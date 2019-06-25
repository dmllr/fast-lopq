#include "include/fast-lopq/searcher.h"

#include <cmath>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <functional>


namespace lopq {

void Searcher::load_model(const std::string& proto_path) {
	model.load(proto_path);
}

float Searcher::distance(const Model::FeatureVector& x, blaze::StaticVector<uint8_t, 2UL> coarse_code, blaze::StaticVector<uint8_t, 16UL>& fine_code, DistanceCache& cache) const {
	float D = 0.0;

	auto& d0 = cache[coarse_code[0]];
	auto& d1 = cache[coarse_code[1]];
	auto d0s = d0.size();

	if (d0s == 0) {
		d0 = model.subquantizer_distances(x, coarse_code, 0);
		d0s = d0.size();
	}
	if (d1.size() == 0)
		d1 = model.subquantizer_distances(x, coarse_code, 1);

	uint32_t c = 0;
	for (auto& e: fine_code) {
		D += (c < d0s) ? d0[c][e] : d1[c - d0s][e];
		c++;
	}

	return D;
}

std::vector<Searcher::Response> Searcher::search(const Model::FeatureVector& x) {
	auto const coarse_code = model.predict_coarse(x);

	return search_in(coarse_code, x);
}

std::vector<Searcher::Response> Searcher::search_in(const Model::CoarseCode& coarse_code, const Model::FeatureVector& x) {
	auto& index = get_cell(coarse_code);
	auto& index_codes = index.vectors;

	if (index_codes.empty())
		return std::vector<Response>();

	DistanceCache distance_cache;

	using i_d = std::pair<uint, float>;
	std::function distance_comparator = [](i_d i1, i_d i2) {
		return i1.second < i2.second;
	};

	std::vector<i_d> distances(index_codes.size());

	// calculate relative distances for all vectors in cluster
	uint32_t c = 0;
	for (auto& e: distances) {
		e.second = distance(x, coarse_code, index_codes[c], distance_cache);
		e.first = c++;
	}

	// sort top N is no filtering required, N*N otherwise
	auto quota = std::min(index.ids.size(), (options.dedup ? options.quota * options.quota : options.quota));
	auto begin = distances.begin();
	auto end = begin + quota;
	std::partial_sort(begin, end, distances.end(), distance_comparator);
	// There is a possible issue, while using dense dataset (having too much duplicates).
	// In this case `sort` instead of `partial_sort` should being used.

	// take top N
	std::vector<Response> top;
	top.reserve(options.quota);
	quota = options.quota;
	auto i = size_t();
	auto distance = 0.0f;
	auto prev_distance = - options.dedup_threshold;
	for (auto it = begin; quota > 0 && it != end; ++it) {
		i = (*it).first;
		distance = (*it).second;
		assert(i < index.ids.size() && " in Searcher::search");
		if (options.dedup && abs(distance - prev_distance) < options.dedup_threshold)
			continue;
		if (options.filtering && !options.filtering_function(index.ids[i], index.metadata[i]))
			continue;
		top.emplace_back(Response(index.ids[i], distance));
		prev_distance = distance;
		quota--;
	}

	return top;
}

} // lopq
