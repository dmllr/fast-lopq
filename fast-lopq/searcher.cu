#include "include/fast-lopq/searcher.cuh"

#include <cmath>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <memory>
#include <cassert>


namespace {

__global__
void all_distances(const lopq::gpu::Model& model, const scalar_t* x_, const size_t sz, const lopq::gpu::Model::Codes& coarse_code, const int n, const uint8_t* vectors_) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = idx; i < n; i += stride) {
		// lopq::gpu::subquantizer_distances(model, x_, sz, coarse_code.x, 0);
		auto distance = 0;
	}
}

} // namespace


namespace lopq {
namespace gpu {

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
		// D += (i < d0s) ? d0[i][e] : d1[i - d0s][e];
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
	
	Searcher::DistanceCache distance_cache;

	using i_d = std::pair<uint, float>;

	std::vector<i_d> distances(cluster_size);

	// calculate relative distances for all vectors in cluster
	all_distances<<<1, 1>>>(model, x_, sz, coarse_code, cluster_size, index_codes_);
	
	// uint32_t c = 0;
	// for (auto& e: distances) {
	// 	e.second = distance(x_, sz, coarse_code, index_codes[c], distance_cache);
	// 	e.first = c++;
	// };

	// // take top N
	// std::partial_sort(
	//		distances.begin(), distances.begin() + quota, distances.end(),
	//		[](i_d i1, i_d i2) {
	//			return i1.second < i2.second;
	//		}
	// );

	std::vector<Searcher::Response> top;

	// assert(quota < distances.size()&&  " in Searcher::search");
	// top.reserve(quota);
	// std::for_each(distances.begin(), std::next(distances.begin(), quota), [&](auto& e) {
	//	assert(e.first < index.ids.size()&&  " in Searcher::search");
	//	top.emplace_back(Response(index.ids[e.first]));
	// });

	return top;
}

} // gpu
} // lopq
