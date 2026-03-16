#pragma once
#include "common.cuh"

void approx_knn_cpu(const PointCloud& pc, int k, int* out_intensities);
void approx_knn_gpu(const PointCloud& h_pc, int k, int* h_out_intensities);