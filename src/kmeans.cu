#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include "kmeans.cuh"

void kmeans_cpu(const PointCloud& pc, int k, int T, int* out_intensities) {
}

void kmeans_gpu(const PointCloud& h_pc, int k, int T, int* h_out_intensities) {
}