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

struct Size {
	int w;
	int h;
};

struct Model final {
	uint32_t num_coarse_splits = 2;
	uint32_t num_fine_splits = 16;

	Model(cublasHandle_t handle);

	struct Codes {
		uint32_t size;
		uint8_t* codes = 0;

		Codes(uint32_t size) : size(size) {
			codes = new uint8_t[size];
		}
	};

	template <class T>
	struct Vector {
		size_t size;
		T* x = 0;

		Vector(size_t size) : size(size) {
			x = new T[size];
		}
	};

	template <class T>
	struct Vector_ {
		size_t size;
		T* x = 0;

		Vector_(size_t size) : size(size) {
			cudaMalloc((void**)&x, size * sizeof(T));
		}

		~Vector_() {
			cudaFree(x);
		}
	};

	void load(const std::string& proto_path);
	Codes predict_coarse(const scalar_t* x, const uint32_t sz) const;
	Codes predict_fine(const scalar_t* x, const uint32_t sz, const Codes& coarse_code) const;

	scalar_t** Cs;
	Size* Cszs;

	scalar_t*** Rs;

	scalar_t*** mus;

	scalar_t*** subquantizers;

private:
	cublasHandle_t handle;

	uint8_t predict_cluster(scalar_t* x, const uint32_t sz, scalar_t* centroids, const uint32_t csz) const;
	Model::Vector_<scalar_t> project(const scalar_t* x, const uint32_t sz, const Codes& coarse_code) const;
};

} // gpu
} // lopq
