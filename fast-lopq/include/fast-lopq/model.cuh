#pragma once

#include <map>
#include <string>
#include <cstdint>
#include <cublas_v2.h>

#ifndef USE_FP32
#define USE_FP32 0
#endif

#if USE_FP32
typedef double scalar_t;
#define cublasgemv cublasDgemv
#else
typedef float scalar_t;
#define cublasgemv cublasSgemv
#endif


namespace lopq {
namespace gpu {

__host__ __device__
struct Model final {
	uint32_t num_coarse_splits = 2;
	uint32_t num_fine_splits = 16;
	uint32_t num_clusters = 0;

	Model(cublasHandle_t handle);

	template <class T>
	struct AbstractVector {
		size_t size;
		T* x = 0;

		AbstractVector() {
			size = 0;
			x = 0;
		}
		T& operator [](int i) {
	        return x[i];
	    }
		const T& operator [](int i) const {
	        return x[i];
	    }
	};

	template <class T>
	struct Vector : public AbstractVector<T> {
		Vector() : AbstractVector<T>() { }
		Vector(size_t size) {
			this->size = size;
			this->x = new T[size];
		}
	};

	struct CUVector : public AbstractVector<scalar_t> {
		CUVector() : AbstractVector() { }
		CUVector(size_t size) {
			this->size = size;
			cudaMalloc((void**)&this->x, size * sizeof(scalar_t));
		}
		void zeros() {
			cudaMemset(x, 0.0, size * sizeof(scalar_t));
		}
		~CUVector() {
			if (x)
				cudaFree(x);
		}
	};

	using Codes = Vector<uint8_t>;
	using SubquantizerDistances = Vector<CUVector>;

	void load(const std::string& proto_path);
	Codes predict_coarse(const scalar_t* x_, const uint32_t sz) const;
	Codes predict_fine(const scalar_t* x_, const uint32_t sz, const Codes& coarse_code) const;
	SubquantizerDistances subquantizer_distances(const scalar_t* x_, const size_t sz, const Codes& coarse_code, uint32_t split) const;

	__host__ __device__
	void subquantizer_distances_dododo(scalar_t* distances_, const scalar_t* x_, const size_t sz, const uint8_t* coarse_code, const uint32_t split) const;

private:
	scalar_t** Cs;
	scalar_t*** Rs;
	scalar_t*** mus;
	scalar_t*** subquantizers;

	cublasHandle_t handle;

	uint8_t predict_cluster(const scalar_t* x, const uint32_t sz, const scalar_t* centroids, const uint32_t csz) const;
	CUVector project(const scalar_t* x, const uint32_t sz, const Codes& coarse_code) const;

	__host__ __device__
	void project_dododo(scalar_t* px_, const scalar_t* x_, const uint32_t sz, const uint8_t* coarse_code) const;
};

} // gpu
} // lopq
