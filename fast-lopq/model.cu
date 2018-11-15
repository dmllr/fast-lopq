#include "include/fast-lopq/model.cuh"
#include "utils/utils.cuh"

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
void project(const Model::Params* model, scalar_t* px_, const scalar_t* x_, const uint32_t sz, const uint8_t* coarse_code_) {
	uint32_t split_size = sz / model->num_coarse_splits;

	scalar_t* r_ = (scalar_t*)malloc(sz * sizeof(scalar_t));
	for (uint32_t split = 0; split < model->num_coarse_splits; ++split) {
		auto& cluster = coarse_code_[split];

		residual<<<1, split_size>>>(&r_[split * split_size], &x_[split * split_size], split_size, cluster, model->Cs[split], model->num_clusters, model->mus[split][cluster]);

		// Can't use cublas[S,D]gemv here, cause of cublas_device library slows down *all* memset and memcpy operations

		// const scalar_t alfa=1.0;
		// const scalar_t beta=0.0;
		// cublasgemv(model.handle, CUBLAS_OP_N, split_size, split_size, &alfa, model.Rs[split][cluster], split_size, &r_[split * split_size], 1, &beta, &px_[split * split_size], 1);

		gemv<<<(split_size / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(model->Rs[split][cluster], &r_[split * split_size], &px_[split * split_size], split_size, split_size);
	}

	free(r_);
}

__device__
void Model::project_dododo(scalar_t* px_, const scalar_t* x_, const uint32_t sz, const uint8_t* coarse_code) const {
	uint32_t split_size = sz / cu->num_coarse_splits;

	scalar_t* r_ = (scalar_t*)malloc(sz * sizeof(scalar_t));
	for (uint32_t split = 0; split < cu->num_coarse_splits; ++split) {
		auto& cluster = coarse_code[split];

		residual<<<1, split_size>>>(&r_[split * split_size], &x_[split * split_size], split_size, cluster, cu->Cs[split], cu->num_clusters, cu->mus[split][cluster]);

		// Can't use cublas[S,D]gemv here, cause of cublas_device library slows down *all* memset and memcpy operations

		// const scalar_t alfa=1.0;
		// const scalar_t beta=0.0;
		// cublasgemv(model.handle, CUBLAS_OP_N, split_size, split_size, &alfa, model.Rs[split][cluster], split_size, &r_[split * split_size], 1, &beta, &px_[split * split_size], 1);

		gemv<<<(split_size / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(cu->Rs[split][cluster], &r_[split * split_size], &px_[split * split_size], split_size, split_size);
	}

	free(r_);
}

__device__
void Model::subquantizer_distances_dododo(scalar_t* distances_, const scalar_t* x_, const size_t sz, const uint8_t* coarse_code, const uint32_t split) const {
	scalar_t* px_ = (scalar_t*)malloc(sz * sizeof(scalar_t));
	memset(px_, 0.0, sz * sizeof(scalar_t));

	project_dododo(px_, x_, sz, coarse_code);

	uint32_t split_size = sz / cu->num_coarse_splits;

	auto sx_ = &px_[split * split_size];  // size = split_size

	uint32_t subsplit_size = split_size / cu->num_fine_splits;

	// scalar_t* distances_ = (scalar_t*)malloc(model.num_fine_splits * model.num_clusters * sizeof(scalar_t));
	memset(distances_, 0.0, cu->num_fine_splits * cu->num_clusters * sizeof(scalar_t));
	
	for (uint32_t subsplit = 0; subsplit < cu->num_fine_splits; ++subsplit) {
		auto fx_ = &sx_[subsplit * subsplit_size];  // size = subsplit_size
		auto ds_ = &distances_[subsplit * cu->num_clusters];

		directions<<<1, cu->num_clusters>>>(fx_, cu->subquantizers[split][subsplit], subsplit_size, ds_);
	}
}

__device__
void subquantizer_distances(const Model::Params* model, scalar_t* distances_, const scalar_t* x_, const size_t sz, const uint8_t* coarse_code_, const uint32_t split) {
	printf("here\n");
	scalar_t* px_ = (scalar_t*)malloc(sz * sizeof(scalar_t));
	memset(px_, 0.0, sz * sizeof(scalar_t));
	printf("here memset\n");

	project(model, px_, x_, sz, coarse_code_);
	printf("here project\n");

	uint32_t split_size = sz / model->num_coarse_splits;

	auto sx_ = &px_[split * split_size];  // size = split_size

	uint32_t subsplit_size = split_size / model->num_fine_splits;

	// scalar_t* distances_ = (scalar_t*)malloc(model.num_fine_splits * model.num_clusters * sizeof(scalar_t));
	memset(distances_, 0.0, model->num_fine_splits * model->num_clusters * sizeof(scalar_t));
	
	for (uint32_t subsplit = 0; subsplit < model->num_fine_splits; ++subsplit) {
		auto fx_ = &sx_[subsplit * subsplit_size];  // size = subsplit_size
		auto ds_ = &distances_[subsplit * model->num_clusters];

		directions<<<1, model->num_clusters>>>(fx_, model->subquantizers[split][subsplit], subsplit_size, ds_);
	}
}


Model::Model(cublasHandle_t handle) : handle(handle) {

}

__global__
void t(const Model::Params* cu) {
	if(threadIdx.x == 0) {
		printf("3. cu->num_coarse_splits %d\n", cu->num_coarse_splits);
		printf("4: ");
		printf("cu->Cs[0][0] %f\n", cu->Cs[0][0]);
		printf(".");
	}
}

__global__
void cudaMallocModel(Model::Params* params, uint32_t num_coarse_splits, uint32_t num_fine_splits, uint32_t num_clusters) {
	params->num_coarse_splits = num_coarse_splits;
	params->num_fine_splits = num_fine_splits;
	params->num_clusters = num_clusters;
	
	params->Cs = (scalar_t**)malloc(num_coarse_splits * sizeof(scalar_t*));

	params->Rs = (scalar_t***)malloc(2 * sizeof(scalar_t**));
	for (uint32_t ri = 0; ri < 2; ++ri)
		params->Rs[ri] = (scalar_t**)malloc(num_clusters * sizeof(scalar_t*));
	
	params->mus = (scalar_t***)malloc(2 * sizeof(scalar_t**));
	for (uint32_t mui = 0; mui < 2; ++mui)
		params->mus[mui] = (scalar_t**)malloc(num_clusters * sizeof(scalar_t*));
	
	params->subquantizers = (scalar_t***)malloc(2 * sizeof(scalar_t**));
	for (uint32_t sui = 0; sui < 2; ++sui)
		params->subquantizers[sui] = (scalar_t**)malloc(num_fine_splits * sizeof(scalar_t*));
	
	printf("1. cu->num_coarse_splits %d\n", params->num_coarse_splits);
	printf("1. cu->num_fine_splits %d\n", params->num_fine_splits);
	printf("1. cu->num_clusters %d\n", params->num_clusters);
}

__global__
void model_setC(Model::Params* cu, scalar_t* C_, uint32_t ci) {
	cu->Cs[ci] = C_;
}

__global__
void model_setR(Model::Params* cu, scalar_t* R_, uint32_t ri, uint32_t cluster) {
	cu->Rs[ri][cluster] = R_;
}

__global__
void model_setMu(Model::Params* cu, scalar_t* mu_, uint32_t mui, uint32_t cluster) {
	cu->Rs[mui][cluster] = mu_;
}

__global__
void model_setSubquantizer(Model::Params* cu, scalar_t* S_, uint32_t sui, uint32_t c) {
	cu->subquantizers[sui][c] = S_;
}

void Model::load(const std::string& proto_path) {
	com::flickr::vision::lopq::LOPQModelParams lopq_params;

	std::ifstream proto_stream(proto_path);
	lopq_params.ParseFromIstream(&proto_stream);

	hu.num_coarse_splits = lopq_params.cs_size();
	hu.num_fine_splits = lopq_params.subs_size() / 2;
	hu.num_clusters = lopq_params.cs(0).shape(0);

	assert(hu.num_coarse_splits);
	assert(hu.num_fine_splits);

	//TODO Check for cuda/cublas response statuses

	cudaMalloc((void**)&cu, sizeof(cu));
	cudaMallocModel<<<1, 1>>>(cu, hu.num_coarse_splits, hu.num_fine_splits, hu.num_clusters);
	cudaCheckError();

	hu.Cs = new scalar_t*[hu.num_coarse_splits];
	for (uint32_t ci = 0; ci < hu.num_coarse_splits; ++ci) {
		const auto& cs = lopq_params.cs(ci);
		auto h = cs.shape(0);
		auto w = cs.shape(1);

		std::cout << "Cs[" << ci << "]: " << h << "x" << w << ((ci == 0) ? "\n" : "");

		scalar_t C[w * h];
		for (uint32_t i = 0; i < h; ++i)
			for (uint32_t j = 0; j < w; ++j)
				C[IDX(i, j, h)] = cs.values(i * w + j);

		auto& C_ = hu.Cs[ci];
		cudaMalloc((void**)&C_, w * h * sizeof(scalar_t));
		cudaMemcpy(C_, C, h * w * sizeof(scalar_t), cudaMemcpyHostToDevice);

		model_setC<<<1, 1>>>(cu, C_, ci);
	}
	cudaCheckError();

	// TODO check 2
	hu.Rs = new scalar_t**[2];
	uint32_t rs_size = lopq_params.rs_size();
	uint32_t rs_half = rs_size / 2;
	for (uint32_t ri = 0; ri < 2; ++ri)
		hu.Rs[ri] = new scalar_t*[rs_half];
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

		auto& R_ = hu.Rs[c / rs_half][c % rs_half];
		cudaMalloc((void**)&R_, w * h * sizeof(*R_));
		cublasSetMatrix(w, h, sizeof(scalar_t), R, w, R_, w);
		
		model_setR<<<1, 1>>>(cu, R_, c / rs_half, c % rs_half);
	}
	cudaCheckError();

	hu.mus = new scalar_t**[2];
	uint32_t mus_size = lopq_params.mus_size();
	uint32_t mus_half = mus_size / 2;
	for (uint32_t mui = 0; mui < 2; ++mui)
		hu.mus[mui] = new scalar_t*[mus_half];
	for (uint32_t c = 0; c < mus_size; ++c) {
		const auto& mu = lopq_params.mus(c);
		auto sz = mu.values_size();
		if (c % mus_half == 0)
			std::cout << '\n';
		std::cout << "\rmu[" << c / mus_half << ", " << c % mus_half << "]: " << sz;

		scalar_t muc[sz];
		for (uint32_t i = 0; i < sz; ++i)
			muc[i] = mu.values(i);

		auto& mu_ = hu.mus[c / mus_half][c % mus_half];
		cudaMalloc((void**)&mu_, sz * sizeof(scalar_t));
		cublasSetVector(sz, sizeof(scalar_t), muc, 1, mu_, 1);
		
		model_setR<<<1, 1>>>(cu, mu_, c / mus_half, c % mus_half);
	}
	cudaCheckError();

	hu.subquantizers = new scalar_t**[2];
	uint32_t subs_size = lopq_params.subs_size();
	uint32_t subs_half = subs_size / 2;
	for (uint32_t si = 0; si < 2; ++si)
		hu.subquantizers[si] = new scalar_t*[subs_half];
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

		auto& S_ = hu.subquantizers[c / subs_half][c % subs_half];
		cudaMalloc((void**)&S_, w * h * sizeof(*S_));
		cublasSetMatrix(w, h, sizeof(scalar_t), S, w, S_, w);
		
		model_setSubquantizer<<<1, 1>>>(cu, S_, c / subs_half, c % subs_half);
	}
	cudaCheckError();
	std::cout << '\n';

	cudaThreadSynchronize();


	// std::cout << '\n';
	// printf("hu.num_coarse_splits %d\n", hu.num_coarse_splits);
	// t<<<1, 1>>>(cu);
	// cudaCheckError();
	// std::cout << '\n';
}

Model::Codes Model::predict_coarse(const scalar_t* x_, const uint32_t sz) const {
	Model::Codes coarse(hu.num_coarse_splits);

	uint32_t split_size = sz / hu.num_coarse_splits;
	for (uint32_t split = 0; split < hu.num_coarse_splits; ++split)
		coarse[split] = predict_cluster(&x_[split * split_size], split_size, hu.Cs[split], hu.num_clusters);

	return coarse;
}

Model::Codes Model::predict_fine(const scalar_t* x_, const uint32_t sz, const Model::Codes& coarse_code) const {
	Model::Codes fine(hu.num_fine_splits);

	auto px_ = project(x_, sz, coarse_code);

	uint32_t split_size = sz / hu.num_coarse_splits;
	for (uint32_t split = 0; split < hu.num_coarse_splits; ++split) {
		// Compute subquantizer codes
		uint32_t subsplit_size = split_size / hu.num_fine_splits;
		for (uint32_t subsplit = 0; subsplit < hu.num_fine_splits; ++subsplit) {
			fine[split * hu.num_fine_splits + subsplit] = predict_cluster(&px_[split * split_size + subsplit * subsplit_size], subsplit_size, hu.subquantizers[split][subsplit], hu.num_clusters);
		}
	}

	return fine;
}

Model::CUVector Model::project(const scalar_t* x_, const uint32_t sz, const Model::Codes& coarse_code) const {
	auto px_ = Model::CUVector(sz);

	uint32_t split_size = sz / hu.num_coarse_splits;

	scalar_t* r_;
	cudaMalloc((void**)&r_, sz * sizeof(r_[0]));
	cudaMemset(r_, 0.0, sz * sizeof(r_[0]));

	for (uint32_t split = 0; split < hu.num_coarse_splits; ++split) {
		auto& cluster = coarse_code[split];

		residual<<<1, split_size>>>(&r_[split * split_size], &x_[split * split_size], split_size, cluster, hu.Cs[split], hu.num_clusters, hu.mus[split][cluster]);

		const scalar_t alfa=1.0;
		const scalar_t beta=0;
		cublasgemv(handle, CUBLAS_OP_N, split_size, split_size, &alfa, hu.Rs[split][cluster], split_size, &r_[split * split_size], 1, &beta, &px_[split * split_size], 1);
	}

	cudaFree(r_);

	return px_;
}

uint8_t Model::predict_cluster(const scalar_t* x, const uint32_t sz, const scalar_t* centroids, const uint32_t csz) const {
	scalar_t* ds_;
	cudaMalloc((void**)&ds_, csz * sizeof(ds_[0]));
	cudaMemset(ds_, 0.0, csz * sizeof(ds_[0]));
	directions<<<1, hu.num_clusters>>>(x, centroids, sz, ds_);
	cudaDeviceSynchronize();
	
	int amin;
	cublasIsamin(handle, csz, ds_, 1, &amin);

	cudaFree(ds_);

	return (uint8_t)(amin - 1);
}

Model::SubquantizerDistances Model::subquantizer_distances(const scalar_t* x_, const size_t sz, const Model::Codes& coarse_code, uint32_t split) const {
	auto px_ = project(x_, sz, coarse_code);

	uint32_t split_size = sz / hu.num_coarse_splits;

	auto sx_ = &px_[split * split_size];  // size = split_size

	uint32_t subsplit_size = split_size / hu.num_fine_splits;

	Model::Vector<Model::CUVector> distances(hu.num_fine_splits);
	
	for (uint32_t subsplit = 0; subsplit < hu.num_fine_splits; ++subsplit) {
		auto fx_ = &sx_[subsplit * subsplit_size];  // size = subsplit_size
		
		CUVector ds(hu.num_clusters);
		ds.zeros();

		directions<<<1, hu.num_clusters>>>(fx_, hu.subquantizers[split][subsplit], subsplit_size, ds.x);

		distances[subsplit] = ds;
	}

	return distances;
}

} // gpu
} // lopq
