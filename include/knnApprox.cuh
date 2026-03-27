#pragma once
#include "common.cuh"

namespace knnApprox {

float knnApproxCPUFloat(const PointCloudFloat &pc, int32_t k, int32_t *newIntensity);
float knnApproxGPUFloat(const PointCloudFloat &h_pc, int32_t k, int32_t *h_newIntensity);
float knnApproxCPUInt(const PointCloudInt &pc, int32_t k, int32_t *newIntensity);
float knnApproxGPUInt(const PointCloudInt &h_pc, int32_t k, int32_t *h_newIntensity);

} // namespace knnApprox