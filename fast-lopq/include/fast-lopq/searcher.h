#pragma once

#include "model.h"

#include <string>
#include <vector>
#include <cstdint>
#include <blaze/Math.h>
#include <unordered_map>


namespace lopq {

struct Searcher {
	constexpr static uint32_t quota = 12;

	struct Cluster final {
		std::vector<std::string> ids;
		std::vector<Model::FineCode> vectors;
	};

	struct Response final {
		Response(const std::string& id)
				: id(id) {
		}

		std::string id;
	};

	void load_model(const std::string& proto_path);

	std::vector<Response> search(const Model::FeatureVector& x);
	std::vector<Response> search_in(const Model::CoarseCode& coarse_code, const Model::FeatureVector& x);

protected:
	virtual Cluster& get_cell(const Model::CoarseCode& coarse_code) = 0;

private:
	Model model;
	std::unordered_map<int, Cluster> clusters;

	using DistanceCache = std::unordered_map<int, blaze::DynamicVector<Model::FloatVector>>;

	float distance(const Model::FeatureVector& x, blaze::StaticVector<uint8_t, 2UL> coarse_code, blaze::StaticVector<uint8_t, 16UL>& fine_code, DistanceCache& cache) const;
};

} // lopq
