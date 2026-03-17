#pragma once
#include "common.cuh"

void knnCPU(const PointCloud& pc, int k, int* newIntensity);
void knnGPU(const PointCloud& h_pc, int k, int* h_newIntensity);