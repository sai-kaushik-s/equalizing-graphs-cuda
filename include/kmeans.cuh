#pragma once
#include "common.cuh"

void kmeans_cpu(const PointCloud& pc, int k, int T, int* out_intensities);
void kmeans_gpu(const PointCloud& h_pc, int k, int T, int* h_out_intensities);