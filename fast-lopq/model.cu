#include "include/fast-lopq/model.cuh"

#include <fstream>
#include <iostream>
#include <limits>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "lopq_model.pb.h"


#define IDX(i, j, s) (((j)*(s))+(i))
#define FINITIALIZER (std::numeric_limits<scalar_t>::infinity())
#define BLOCK_SIZE 32

namespace {

void log1d(const std::string& name, scalar_t* x, int w) {
	scalar_t* x_ = new scalar_t[w];
	cudaMemcpy(x_, x, w * sizeof(scalar_t), cudaMemcpyDeviceToHost);

	std::cout << name << "\n";
	for (int i = 0; i < w; ++i) {
		std::cout << x_[i] << ", ";
		if ((i+1) % 5 == 0)
			std::cout << '\n';
	}
	std::cout << '\n';

	delete[] x_;
}

void log2d(const std::string& name, scalar_t* x_, int d) {
	scalar_t* x = new scalar_t[256*64];
	cudaMemcpy(x, x_, 256*64 * sizeof(scalar_t), cudaMemcpyDeviceToHost);

//	std::cout << name << "\n";
//	for (int i = 0; i < 4; ++i) {
//		for (int j = 0; j < 4; ++j)
//			std::cout << x[IDX(i, j, d)] << ", ";
//		std::cout << '\n';
//	}
//	std::cout << '\n';


	std::cout << name << " samples\n";
	std::cout << 64 << "-" << 0 << ": " << x[IDX(64, 0, d)] << "\n";
	std::cout << 64 << "-" << 1 << ": " << x[IDX(64, 1, d)] << "\n";

	delete[] x;
}

} // namespace


namespace lopq {
namespace gpu {

__global__
void directions(const scalar_t* x_, const scalar_t* C_, uint8_t cszw, scalar_t* ds_) {
	for (int i = 0; i < cszw; ++i) {
		auto v = static_cast<scalar_t>(x_[i] - C_[IDX(threadIdx.x, i, blockDim.x)]);
		ds_[threadIdx.x] += v * v;
	}
}

__global__
void residual(scalar_t* r_, const scalar_t* x_, uint8_t sz, const uint8_t cluster, const scalar_t* C_, const int csz, const scalar_t* mu_) {
	auto i = threadIdx.x;
	r_[i] = static_cast<scalar_t>(x_[i] - C_[IDX(cluster, i, csz)] - mu_[i]);
}

__global__
void gemv(const scalar_t* __restrict__ A_, const scalar_t* __restrict__ x_, scalar_t* __restrict__ y_, const uint32_t h, const uint32_t w) {
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ scalar_t shx_[BLOCK_SIZE];

	scalar_t yv_ = 0.0;

	#pragma unroll
	for (unsigned int m = 0; m < ((w + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m) {
		if ((m * BLOCK_SIZE + threadIdx.x) <  w)
			shx_[threadIdx.x] = x_[threadIdx.x + m * BLOCK_SIZE];
		else
			shx_[threadIdx.x] = 0.f;

		__syncthreads();

		#pragma unroll
		for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
			// column-major ordering
			yv_ += A_[tid + (e + BLOCK_SIZE * m) * h] * shx_[e];

			// row-major ordering
			//yv_ += A_[tid * w + (e + BLOCK_SIZE * m)] * shx_[e];
		}

		__syncthreads();
	}

	if (tid < h)
		y_[tid] = yv_;
}

__device__
void project(const Model& model, scalar_t* px_, const scalar_t* x_, const uint32_t sz, const uint8_t* coarse_code) {
	uint32_t split_size = sz / model.num_coarse_splits;

	scalar_t* r_ = (scalar_t*)malloc(sz);
	for (uint32_t split = 0; split < model.num_coarse_splits; ++split) {
		auto& cluster = coarse_code[split];

		residual<<<1, split_size>>>(&r_[split * split_size], &x_[split * split_size], split_size, cluster, model.Cs[split], model.num_clusters, model.mus[split][cluster]);

		// cublas_device library slows down all memset and memcpy operation
		// const scalar_t alfa=1.0;
		// const scalar_t beta=0.0;
		// cublasgemv(model.handle, CUBLAS_OP_N, split_size, split_size, &alfa, model.Rs[split][cluster], split_size, &r_[split * split_size], 1, &beta, &px_[split * split_size], 1);
		gemv<<<(split_size / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(model.Rs[split][cluster], &r_[split * split_size], &px_[split * split_size], split_size, split_size);
	}

	free(r_);
}

__device__
void subquantizer_distances(const Model& model, scalar_t* distances_, const scalar_t* x_, const size_t sz, const uint8_t* coarse_code, const uint32_t split) {
	scalar_t* px_ = (scalar_t*)malloc(sz);
	memset(px_, 0.0, sz * sizeof(scalar_t));

	project(model, px_, x_, sz, coarse_code);

	uint32_t split_size = sz / model.num_coarse_splits;

	auto sx_ = &px_[split * split_size];  // size = split_size

	uint32_t subsplit_size = split_size / model.num_fine_splits;

	// scalar_t* distances_ = (scalar_t*)malloc(model.num_fine_splits * model.num_clusters);
	memset(distances_, 0.0, model.num_fine_splits * model.num_clusters * sizeof(scalar_t));
	
	for (uint32_t subsplit = 0; subsplit < model.num_fine_splits; ++subsplit) {
		auto fx_ = &sx_[subsplit * subsplit_size];  // size = subsplit_size
		auto ds_ = &distances_[subsplit * model.num_clusters];

		directions<<<1, model.num_clusters>>>(fx_, model.subquantizers[split][subsplit], subsplit_size, ds_);
	}
}


Model::Model(cublasHandle_t handle) : handle(handle) {

}

void Model::load(const std::string& proto_path) {
	com::flickr::vision::lopq::LOPQModelParams lopq_params;

	std::ifstream proto_stream(proto_path);
	lopq_params.ParseFromIstream(&proto_stream);

	num_coarse_splits = lopq_params.cs_size();
	num_fine_splits = lopq_params.subs_size() / 2;

	assert(num_coarse_splits);
	assert(num_fine_splits);

	//TODO Check for cuda/cublas response statuses

	num_clusters = lopq_params.cs(0).shape(0);

	Cs = new scalar_t*[num_coarse_splits];
	for (uint32_t ci = 0; ci < num_coarse_splits; ++ci) {
		const auto& cs = lopq_params.cs(ci);
		auto h = cs.shape(0);
		auto w = cs.shape(1);

		std::cout << "Cs[" << ci << "]: " << h << "x" << w << '\n';

		scalar_t C[w * h];
		for (uint32_t i = 0; i < h; ++i)
			for (uint32_t j = 0; j < w; ++j)
				C[IDX(i, j, h)] = cs.values(i * w + j);

		cudaMalloc((void**)&Cs[ci], w * h * sizeof(scalar_t));
		cudaMemset(Cs[ci], FINITIALIZER, h * w * sizeof(scalar_t));
		cudaMemcpy(Cs[ci], C, h * w * sizeof(scalar_t), cudaMemcpyHostToDevice);
	}

	Rs = new scalar_t**[2];
	uint32_t rs_size = lopq_params.rs_size();
	uint32_t rs_half = rs_size / 2;
	for (uint32_t ri = 0; ri < 2; ++ri)
		Rs[ri] = new scalar_t*[rs_half];
	for (uint32_t c = 0; c < rs_size; ++c) {
		const auto& rs = lopq_params.rs(c);

		auto h = rs.shape(0);
		auto w = rs.shape(1);
		if (c % rs_half == 0)
			std::cout << '\n';
		std::cout << "\rRs[" << c / rs_half << ", " << c % rs_half << "]: " << h << "x" << w;

		scalar_t R[w * h];
		for (uint32_t i = 0; i < h; ++i) {
			for (uint32_t j = 0; j < w; ++j)
				R[IDX(i, j, h)] = rs.values(i * w + j);
		}

		auto& R_ = Rs[c / rs_half][c % rs_half];
		cudaMalloc((void**)&R_, w * h * sizeof(*R_));
		cublasSetMatrix(w, h, sizeof(scalar_t), R, w, R_, w);
	}
	std::cout << '\n';

	mus = new scalar_t**[2];
	uint32_t mus_size = lopq_params.mus_size();
	uint32_t mus_half = mus_size / 2;
	for (uint32_t mui = 0; mui < 2; ++mui)
		mus[mui] = new scalar_t*[mus_half];
	for (uint32_t c = 0; c < mus_size; ++c) {
		const auto& mu = lopq_params.mus(c);
		auto sz = mu.values_size();
		if (c % mus_half == 0)
			std::cout << '\n';
		std::cout << "\rmu[" << c / mus_half << ", " << c % mus_half << "]: " << sz;

		scalar_t muc[sz];
		for (uint32_t i = 0; i < sz; ++i)
			muc[i] = mu.values(i);

		auto& mu_ = mus[c / mus_half][c % mus_half];
		cudaMalloc((void**)&mu_, sz * sizeof(scalar_t));
		cudaMemset(mu_, FINITIALIZER, sz * sizeof(scalar_t));
		cublasSetVector(sz, sizeof(scalar_t), muc, 1, mu_, 1);
	}
	std::cout << '\n';

	subquantizers = new scalar_t**[2];
	uint32_t subs_size = lopq_params.subs_size();
	uint32_t subs_half = subs_size / 2;
	for (uint32_t si = 0; si < 2; ++si)
		subquantizers[si] = new scalar_t*[subs_half];
	for (uint32_t c = 0; c < subs_size; ++c) {
		const auto& subs = lopq_params.subs(c);

		auto h = subs.shape(0);
		auto w = subs.shape(1);
		if (c % subs_half == 0)
			std::cout << '\n';
		std::cout << "\rsubquantizers[" << c / subs_half << ", " << c % subs_half << "]: " << h << "x" << w;

		scalar_t S[w * h];
		for (uint32_t i = 0; i < h; ++i) {
			for (uint32_t j = 0; j < w; ++j)
				S[IDX(i, j, h)] = subs.values(i * w + j);
		}

		auto& S_ = subquantizers[c / subs_half][c % subs_half];
		cudaMalloc((void**)&S_, w * h * sizeof(*S_));
		cublasSetMatrix(w, h, sizeof(scalar_t), S, w, S_, w);
	}
	std::cout << '\n';
}

Model::Codes Model::predict_coarse(const scalar_t* x_, const uint32_t sz) const {
	Model::Codes coarse(num_coarse_splits);

	uint32_t split_size = sz / num_coarse_splits;
	for (uint32_t split = 0; split < num_coarse_splits; ++split)
		coarse[split] = predict_cluster(&x_[split * split_size], split_size, Cs[split], num_clusters);

	return coarse;
}

Model::Codes Model::predict_fine(const scalar_t* x_, const uint32_t sz, const Model::Codes& coarse_code) const {
	Model::Codes fine(num_fine_splits);

	auto px_ = project(x_, sz, coarse_code);

	uint32_t split_size = sz / num_coarse_splits;
	for (uint32_t split = 0; split < num_coarse_splits; ++split) {
		// Compute subquantizer codes
		uint32_t subsplit_size = split_size / num_fine_splits;
		for (uint32_t subsplit = 0; subsplit < num_fine_splits; ++subsplit) {
			fine[split * num_fine_splits + subsplit] = predict_cluster(&px_[split * split_size + subsplit * subsplit_size], subsplit_size, subquantizers[split][subsplit], num_clusters);
		}
	}

	return fine;
}

Model::CUVector Model::project(const scalar_t* x_, const uint32_t sz, const Model::Codes& coarse_code) const {
	auto px_ = Model::CUVector(sz);

	uint32_t split_size = sz / num_coarse_splits;

	scalar_t* r_;
	cudaMalloc((void**)&r_, sz * sizeof(r_[0]));
	cudaMemset(r_, 0.0, sz * sizeof(r_[0]));

	for (uint32_t split = 0; split < num_coarse_splits; ++split) {
		auto& cluster = coarse_code[split];

		residual<<<1, split_size>>>(&r_[split * split_size], &x_[split * split_size], split_size, cluster, Cs[split], num_clusters, mus[split][cluster]);

		const scalar_t alfa=1.0;
		const scalar_t beta=0;
		cublasgemv(handle, CUBLAS_OP_N, split_size, split_size, &alfa, Rs[split][cluster], split_size, &r_[split * split_size], 1, &beta, &px_[split * split_size], 1);
	}

	cudaFree(r_);

	return px_;
}

uint8_t Model::predict_cluster(const scalar_t* x, const uint32_t sz, const scalar_t* centroids, const uint32_t csz) const {
	scalar_t* ds_;
	cudaMalloc((void**)&ds_, csz * sizeof(ds_[0]));
	cudaMemset(ds_, 0.0, csz * sizeof(ds_[0]));
	directions<<<1, num_clusters>>>(x, centroids, sz, ds_);
	cudaDeviceSynchronize();
	
	int amin;
	cublasIsamin(handle, csz, ds_, 1, &amin);

	cudaFree(ds_);

	return (uint8_t)(amin - 1);
}

Model::SubquantizerDistances Model::subquantizer_distances(const scalar_t* x_, const size_t sz, const Model::Codes& coarse_code, uint32_t split) const {
	auto px_ = project(x_, sz, coarse_code);

	uint32_t split_size = sz / num_coarse_splits;

	auto sx_ = &px_[split * split_size];  // size = split_size

	uint32_t subsplit_size = split_size / num_fine_splits;

	Model::Vector<Model::CUVector> distances(num_fine_splits);
	
	for (uint32_t subsplit = 0; subsplit < num_fine_splits; ++subsplit) {
		auto fx_ = &sx_[subsplit * subsplit_size];  // size = subsplit_size
		
		CUVector ds(num_clusters);
		ds.zeros();

		directions<<<1, num_clusters>>>(fx_, subquantizers[split][subsplit], subsplit_size, ds.x);

		distances[subsplit] = ds;
	}

	return distances;
}

} // gpu
} // lopq
