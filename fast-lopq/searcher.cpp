#include "include/fast-lopq/searcher.h"

#include <fstream>
#include <algorithm>
#include <cassert>
#include <queue>


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

	auto c = size_t(0);
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

	auto distance_cache = DistanceCache();

	using i_d = std::pair<float, uint>;

	auto distances = std::vector<i_d>(index_codes.size());

	// calculate relative distances for all vectors in cluster
	auto c = size_t(0);
	for (auto& e: distances) {
		e.first = distance(x, coarse_code, index_codes[c], distance_cache);
		e.second = c++;
	}

    // priority queue for top results fetching
	auto distance_queue = std::priority_queue<i_d, decltype(distances), std::greater<>>(std::greater<>(), std::move(distances));

    auto top = std::vector<Response>();
    top.reserve(options.quota);

    auto offset = options.offset;
    auto quota = std::min(index.ids.size() - options.offset, options.quota);
    auto i = size_t();
    auto distance = 0.0f;
    auto prev_distance = -options.dedup_threshold;
    while (!distance_queue.empty() && (quota + offset > 0)) {
        auto& it = distance_queue.top();
        distance = it.first;
        i = it.second;
        distance_queue.pop();

        assert(i < index.ids.size() && " in Searcher::search");

        if (options.dedup && abs(distance - prev_distance) < options.dedup_threshold)
            continue;
        if (options.filtering && !options.filtering_function(index.ids[i], index.metadata[i]))
            continue;
        if (offset > 0) {
            offset--;
            prev_distance = distance;
            continue;
        }

        top.emplace_back(Response(index.ids[i], distance));
        prev_distance = distance;
        quota--;
    }

	return top;
}

} // lopq
