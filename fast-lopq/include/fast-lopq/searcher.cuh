#pragma once

#include "model.cuh"

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include <cublas_v2.h>


namespace lopq {
namespace gpu {

struct Searcher {
	// inline static uint32_t quota = 12;

	Searcher(cublasHandle_t handle);

	struct Cluster final {
		std::vector<std::string> ids;
		std::vector<Model::Codes> vectors;
	};

	struct Response final {
		Response(const std::string& id)
				: id(id) {
		}

		std::string id;
	};

	void load_model(const std::string& proto_path);

	std::vector<Response> search(const scalar_t* x_);
	std::vector<Response> search_in(const Model::Codes& coarse_code, const scalar_t* x_, const size_t sz);

protected:
	virtual Cluster& get_cell(const Model::Codes& coarse_code) = 0;

private:
	Model model = 0;
	std::unordered_map<int, Cluster> clusters;

	cublasHandle_t handle;

	using DistanceCache = std::unordered_map<int, Model::SubquantizerDistances>;

	float distance(const scalar_t* x, const size_t sz, const Model::Codes& coarse_code, const Model::Codes& fine_code, Searcher::DistanceCache& cache) const;
};

} // gpu
} // lopq
