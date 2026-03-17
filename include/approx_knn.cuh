#pragma once
#include "common.cuh"

void knnApproxCPU(const PointCloud& pc, int k, int* newIntensity);
void knnApproxGPU(const PointCloud& h_pc, int k, int* h_newIntensity);