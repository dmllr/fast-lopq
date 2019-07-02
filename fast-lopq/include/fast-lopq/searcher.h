#include <utility>

#pragma once

#include "model.h"

#include <string>
#include <vector>
#include <cstdint>
#include <blaze/Math.h>
#include <unordered_map>
#include <functional>


namespace lopq {

struct Searcher {
	struct Cluster final {
		std::vector<std::string> ids;
		std::vector<Model::FineCode> vectors;
		std::vector<std::string> metadata;
	};

	struct Options final {
		using FilteringFunction = std::function<bool(const std::string& id, const std::string& meta)>;

		Options& limit(uint32_t q) {
			quota = q;
			return *this;
		}

		Options& deduplication() {
			dedup = true;
			return *this;
		}

		Options& no_deduplication() {
			dedup = false;
			return *this;
		}

		Options& deduplication(float threshold) {
			this->dedup = true;
			this->dedup_threshold = threshold;
			return *this;
		}

		Options& filter(const FilteringFunction& f_runnable) {
			filtering = true;
			filtering_function = f_runnable;
			return *this;
		}

		size_t quota = 12;
		bool dedup = false;
		float dedup_threshold = 0.0001;
		bool filtering = false;
		FilteringFunction filtering_function;
	};

	Options& configure() {
		return options;
	}

	struct Response final {
		explicit Response(std::string id)
			: id(std::move(id)), distance(0) { }

		explicit Response(std::string id, float distance)
			: id(std::move(id)), distance(distance) { }

		std::string id;
		float distance;
	};

	void load_model(const std::string& proto_path);

	std::vector<Response> search(const Model::FeatureVector& x);
	std::vector<Response> search_in(const Model::CoarseCode& coarse_code, const Model::FeatureVector& x);

protected:
	virtual Cluster& get_cell(const Model::CoarseCode& coarse_code) = 0;

private:
	Model model;
	std::unordered_map<int, Cluster> clusters;
	Options options;

	using DistanceCache = std::unordered_map<int, blaze::DynamicVector<Model::FloatVector>>;

	float distance(const Model::FeatureVector& x, blaze::StaticVector<uint8_t, 2UL> coarse_code, blaze::StaticVector<uint8_t, 16UL>& fine_code, DistanceCache& cache) const;
};

} // lopq
