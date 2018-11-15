#pragma once

#include <cuda_runtime.h>


#define cudaSafe(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)


inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA call failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

inline void __cudaCheckError(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	err = cudaDeviceSynchronize();
	if(cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}
