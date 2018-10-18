#include "include/fast-lopq/model.cuh"

#include <fstream>
#include <iostream>
#include <limits>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "lopq_model.pb.h"


#define IDX(i, j, s) (((j)*(s))+(i))
#define FINITIALIZER (std::numeric_limits<scalar_t>::infinity())


namespace {

template <class T>
void log1d(const std::string& name, T* x, int w) {
	T* x_ = new T[w];
	cudaMemcpy(x_, x, w * sizeof(T), cudaMemcpyDeviceToHost);

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

	delete x;
}

__global__
void susq(const scalar_t* x_, const scalar_t* C_, uint8_t cszw, scalar_t* ds_) {
	for (int i = 0; i < cszw; ++i) {
		auto v = static_cast<scalar_t>(x_[i] - C_[IDX(threadIdx.x, i, blockDim.x)]);

		ds_[threadIdx.x] += v * v;
	}
}

__global__
void residual(scalar_t* r_, const scalar_t* x_, uint8_t sz, const uint8_t cluster, const scalar_t* C_, const int csz, const scalar_t* mu_) {
	for (int i = 0; i < sz; ++i)
		r_[i] = static_cast<scalar_t>(x_[i] - C_[IDX(cluster, i, csz)] - mu_[i]);
}

} // namespace


namespace lopq {
namespace gpu {

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
		// cudaMemcpy(mu_, muc, sz * sizeof(scalar_t), cudaMemcpyHostToDevice);

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

Model::Codes Model::predict_coarse(const scalar_t* x, const uint32_t sz) const {
	scalar_t* x_;
	cudaMalloc((void**)&x_, sz * sizeof(scalar_t));
	cublasSetVector(sz, sizeof(scalar_t), x, 1, x_, 1);

	Model::Codes coarse(num_coarse_splits);

	uint32_t split_size = sz / num_coarse_splits;
	for (uint32_t split = 0; split < num_coarse_splits; ++split)
		coarse[split] = predict_cluster(&x_[split * split_size], split_size, Cs[split], num_clusters);

	return coarse;
}

Model::Codes Model::predict_fine(const scalar_t* x, const uint32_t sz, const Model::Codes& coarse_code) const {
	scalar_t* x_;
	cudaMalloc((void**)&x_, sz * sizeof(scalar_t));
	cublasSetVector(sz, sizeof(scalar_t), x, 1, x_, 1);

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

		residual<<<1, 1>>>(&r_[split * split_size], &x_[split * split_size], split_size, cluster, Cs[split], num_clusters, mus[split][cluster]);

		const scalar_t alfa=1.0;
		const scalar_t beta=0;
		cublasgemv(handle, CUBLAS_OP_N, split_size, split_size, &alfa, Rs[split][cluster], split_size, &r_[split * split_size], 1, &beta, &px_[split * split_size], 1);
	}

	cudaFree(r_);

	return px_;
}

uint8_t Model::predict_cluster(scalar_t* x, const uint32_t sz, scalar_t* centroids, const uint32_t csz) const {
	scalar_t* ds_;
	cudaMalloc((void**)&ds_, csz * sizeof(ds_[0]));
	cudaMemset(ds_, 0.0, csz * sizeof(ds_[0]));
	susq<<<1, csz>>>(x, centroids, sz, ds_);
	cudaDeviceSynchronize();

	int amin;
	cublasIsamin(handle, csz, ds_, 1, &amin);

	cudaFree(ds_);

	return (uint8_t)(amin - 1);
}

} // gpu
} // lopq
