#include "include/fast-lopq/searcher.cuh"

#include <cmath>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <memory>
#include <cassert>

#include <iostream>

#define BLOCK_SIZE 32

namespace lopq {
namespace gpu {

__device__
scalar_t distance(const lopq::gpu::Model& model, const scalar_t* x_, const size_t sz, const lopq::gpu::Model::Codes& coarse_code, const uint8_t* fine_code_) {
	scalar_t D = 99.55;

	printf("3. model.num_fine_splits, model.num_clusters: %d, %d\n", model.num_fine_splits, model.num_clusters);

	printf("d0\n");
	// scalar_t* d0_ = malloc(model.num_fine_splits * model.num_clusters * sizeof(scalar_t));
	// auto d0_ = new scalar_t[model.num_fine_splits * model.num_clusters];
	printf("%d, %d\n", model.num_fine_splits, model.num_clusters);
	scalar_t d0_[8 * 256];
	printf("malloc yes\n");
	model.subquantizer_distances_dododo(d0_, x_, sz, coarse_code.x, 0);
	printf("subq yes\n");
	auto d0s = 128;  // TODO replace 128
	printf("d0 yes\n");

	// printf("d1\n");
	// scalar_t* d1_ = (scalar_t*)malloc(model.num_fine_splits * model.num_clusters * sizeof(scalar_t));
	// lopq::gpu::subquantizer_distances(model, d1_, x_, sz, coarse_code.x, 1);
	// printf("d1 yes\n");

	// for (uint32_t i = 0; i < model.num_fine_splits; ++i) {
	// 	auto& e = fine_code_[i];
	// 	D += (i < d0s) ? d0_[i * model.num_clusters + e] : d1_[(i - d0s) * model.num_clusters + e];
	// };
	// printf("loop yes\n");
	
	// free(d0_);
	// free(d1_);
	printf("free yes\n");

	return D;
}

__global__
void all_distances(const lopq::gpu::Model& model, const scalar_t* x_, const size_t sz, const lopq::gpu::Model::Codes& coarse_code, const int n, const uint8_t* vectors_, scalar_t* distances_) {
	printf("2. model.num_fine_splits, model.num_clusters: %d, %d\n", model.num_fine_splits, model.num_clusters);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += stride) {
		auto fine_code_ = &vectors_[i * 16];  // TODO replace 16
		distances_[i] = distance(model, x_, sz, coarse_code, fine_code_);
	}
}

Searcher::Searcher(cublasHandle_t handle) : handle(handle) {
	Model m(handle);
	model = m;
}

void Searcher::load_model(const std::string& proto_path) {
	model.load(proto_path);
}

scalar_t Searcher::distance(const scalar_t* x_, const size_t sz, const Model::Codes& coarse_code, const Model::Codes& fine_code, Searcher::DistanceCache& cache) const {
	scalar_t D = 0.0;

	auto& d0 = cache[coarse_code[0]];
	auto& d1 = cache[coarse_code[1]];
	auto d0s = d0.size;

	if (d0s == 0) {
		d0 = model.subquantizer_distances(x_, sz, coarse_code, 0);
		d0s = d0.size;
	}
	if (d1.size == 0)
		d1 = model.subquantizer_distances(x_, sz, coarse_code, 1);

	for (uint32_t i = 0; i < model.num_fine_splits; ++i) {
		auto& e = fine_code[i];
		D += (i < d0s) ? d0[i][e] : d1[i - d0s][e];
	};

	return D;
}

std::vector<Searcher::Response> Searcher::search(const scalar_t* x_) {
	auto coarse_code = model.predict_coarse(x_, 128);

	return search_in(coarse_code, x_, 128);
}

std::vector<Searcher::Response> Searcher::search_in(const Model::Codes& coarse_code, const scalar_t* x_, const size_t sz) {
	auto& index = get_cell(coarse_code);

	auto cluster_size = index.ids.size();
	const auto& index_codes_ = index.vectors;

	if (cluster_size == 0)
		return std::vector<Response>();
	
	// Searcher::DistanceCache distance_cache;

	using i_d = std::pair<uint, float>;

	std::vector<i_d> distances(cluster_size);

	// std::cout << "cluster_size: " << cluster_size << "\n";
	printf("1. model.num_fine_splits, model.num_clusters: %d, %d\n", model.num_fine_splits, model.num_clusters);

	// calculate relative distances for all vectors in cluster
	scalar_t* distances_;
	cudaMalloc((void**)&distances_, cluster_size * sizeof(scalar_t));
	cudaMemset(distances_, 0, cluster_size * sizeof(scalar_t));

	// all_distances<<<(cluster_size / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(model, x_, sz, coarse_code, cluster_size, index_codes_, distances_);
	all_distances<<<1, 1>>>(model, x_, sz, coarse_code, cluster_size, index_codes_, &distances_[0]);

	auto ldistances = new scalar_t[cluster_size];
	for (int i = 0; i < 4; ++i)
		ldistances[i] = 11.99;

	cudaMemcpy(ldistances, distances_, cluster_size * sizeof(scalar_t), cudaMemcpyDeviceToHost);

	// std::cout << "ldistances: ";
	// for (int i = 0; i < 4; ++i)
	// 	std::cout << ldistances[i] << " ";
	// std::cout << "\n";

	uint32_t c = 0;
	for (auto& e: distances) {
		e.second = ldistances[c];
		e.first = c++;
	};
	
	delete[] ldistances;

	 // take top N
	std::partial_sort(
			distances.begin(), distances.begin() + 12, distances.end(),
			[](i_d i1, i_d i2) {
				return i1.second < i2.second;
			}
	);

	std::vector<Searcher::Response> top;

	top.reserve(12);

	for(int i = 0; i < 12; ++i) {
		top.emplace_back(Response(index.ids[distances[i].first]));
	}

	return top;
}

} // gpu
} // lopq
