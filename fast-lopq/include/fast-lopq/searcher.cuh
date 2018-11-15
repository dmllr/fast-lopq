#pragma once

#include "model.cuh"

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cublas_v2.h>


namespace lopq {
namespace gpu {

__host__ __device__
struct Searcher {
	// inline static uint32_t quota = 12;
	Model model = 0;

	Searcher(cublasHandle_t handle);

	struct Cluster final {
		std::vector<std::string> ids;
		uint8_t* vectors;
		uint32_t size;
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
	std::unordered_map<int, Cluster> clusters;

	cublasHandle_t handle;

	using DistanceCache = std::unordered_map<int, Model::SubquantizerDistances>;

	scalar_t distance(const scalar_t* x, const size_t sz, const Model::Codes& coarse_code, const Model::Codes& fine_code, Searcher::DistanceCache& cache) const;
};

} // gpu
} // lopq
