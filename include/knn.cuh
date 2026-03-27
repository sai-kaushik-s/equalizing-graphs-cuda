#pragma once
#include "common.cuh"

namespace knn {

float knnCPUFloat(const PointCloudFloat &pc, int32_t k, int32_t *newIntensity);
float knnGPUFloat(const PointCloudFloat &h_pc, int32_t k, int32_t *h_newIntensity);
float knnCPUInt(const PointCloudInt &pc, int32_t k, int32_t *newIntensity);
float knnGPUInt(const PointCloudInt &h_pc, int32_t k, int32_t *h_newIntensity);

} // namespace knn