#pragma once
#include "common.cuh"

void knn_cpu(const PointCloud& pc, int k, int* out_intensities);
void knn_gpu(const PointCloud& h_pc, int k, int* h_out_intensities);