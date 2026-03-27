#pragma once
#include "common.cuh"

namespace kMeans {

float kMeansCPUFloat(const PointCloudFloat &pc, int32_t k, int32_t T, int32_t *newIntensity);
float kMeansGPUFloat(const PointCloudFloat &h_pc, int32_t k, int32_t T, int32_t *h_newIntensity);
float kMeansCPUInt(const PointCloudInt &pc, int32_t k, int32_t T, int32_t *newIntensity);
float kMeansGPUInt(const PointCloudInt &h_pc, int32_t k, int32_t T, int32_t *h_newIntensity);

} // namespace kMeans