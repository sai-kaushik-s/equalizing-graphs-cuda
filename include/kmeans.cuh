#pragma once
#include "common.cuh"

void kMeansCPU(const PointCloud& pc, int k, int T, int* newIntensity);
void kMeansGPU(const PointCloud& h_pc, int k, int T, int* h_newIntensity);