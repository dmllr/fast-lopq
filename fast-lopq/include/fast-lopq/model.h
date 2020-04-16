#pragma once

#include <map>
#include <string>
#include <cstdint>
#include <blaze/Math.h>


namespace lopq {

struct Model final {
	uint32_t num_coarse_splits = 2;
	uint32_t num_fine_splits = 16;

	using FloatMatrix = blaze::DynamicMatrix<float>;
	using FloatVector = blaze::DynamicVector<float>;
	using FeatureVector = blaze::DynamicVector<double>;
	using CoarseCode = blaze::StaticVector<uint8_t, 2UL>;
	using FineCode = blaze::StaticVector<uint8_t, 16UL>;
	using CValues = blaze::StaticVector<FloatMatrix, 2UL>;
	using RValues = blaze::StaticVector<blaze::DynamicVector<FloatMatrix>, 2UL>;
	using MuValues = blaze::StaticVector<blaze::DynamicVector<FloatVector>, 2UL>;
	using SQValues = blaze::StaticVector<blaze::DynamicVector<FloatMatrix>, 2UL>;

	void load(const std::string& proto_path);
	CoarseCode predict_coarse(const FeatureVector& x) const;
	FineCode predict_fine(const FeatureVector& x, const CoarseCode& coarse_codes) const;
	blaze::DynamicVector<FloatVector> subquantizer_distances(const FeatureVector& x, const CoarseCode& coarse_codes, uint32_t split) const;

private:
	CValues Cs;
	RValues Rs;
	MuValues mus;
	SQValues subquantizers;

	const blaze::StaticVector<double, 128UL> project(const FeatureVector& x, const CoarseCode& coarse_codes) const;
};

} // lopq
