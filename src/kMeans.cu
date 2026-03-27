#include "kMeans.cuh"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

namespace kMeans {

struct CentroidInt {
    int32_t x, y, z;
};

struct CentroidFloat {
    float x, y, z;
};

void assignClustersCPUInt(const PointCloudInt &pc, const std::vector<CentroidInt> &centroids,
                          int32_t *clusterAssignments) {
    int32_t k = static_cast<int32_t>(centroids.size());
#pragma omp parallel for
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        int32_t minDist = std::numeric_limits<int32_t>::max();
        int32_t bestCluster = 0;
        for (int32_t j = 0; j < k; ++j) {
            int32_t dist = euclideanDistanceCPUInt(pc.x[i], pc.y[i], pc.z[i], centroids[j].x,
                                                   centroids[j].y, centroids[j].z);
            bool closer = dist < minDist;
            if (!closer && dist == minDist) {
                if (centroids[j].x < centroids[bestCluster].x)
                    closer = true;
                else if (centroids[j].x == centroids[bestCluster].x) {
                    if (centroids[j].y < centroids[bestCluster].y)
                        closer = true;
                    else if (centroids[j].y == centroids[bestCluster].y &&
                             centroids[j].z < centroids[bestCluster].z)
                        closer = true;
                }
            }
            if (closer) {
                minDist = dist;
                bestCluster = j;
            }
        }
        clusterAssignments[i] = bestCluster;
    }
}

void assignClustersCPUFloat(const PointCloudFloat &pc, const std::vector<CentroidFloat> &centroids,
                            int32_t *clusterAssignments) {
    int32_t k = static_cast<int32_t>(centroids.size());
#pragma omp parallel for
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        float minDist = std::numeric_limits<float>::max();
        int32_t bestCluster = 0;
        for (int32_t j = 0; j < k; ++j) {
            float dist = euclideanDistanceCPUFloat(pc.x[i], pc.y[i], pc.z[i], centroids[j].x,
                                                   centroids[j].y, centroids[j].z);
            bool closer = dist < minDist;
            if (!closer && dist == minDist) {
                if (centroids[j].x < centroids[bestCluster].x)
                    closer = true;
                else if (centroids[j].x == centroids[bestCluster].x) {
                    if (centroids[j].y < centroids[bestCluster].y)
                        closer = true;
                    else if (centroids[j].y == centroids[bestCluster].y &&
                             centroids[j].z < centroids[bestCluster].z)
                        closer = true;
                }
            }
            if (closer) {
                minDist = dist;
                bestCluster = j;
            }
        }
        clusterAssignments[i] = bestCluster;
    }
}

void updateCentroidsCPUInt(const PointCloudInt &pc, std::vector<CentroidInt> &centroids,
                           const int32_t *clusterAssignments) {
    int32_t k = static_cast<int32_t>(centroids.size());
    std::vector<int64_t> sumX(k, 0), sumY(k, 0), sumZ(k, 0);
    std::vector<int32_t> counts(k, 0);
#pragma omp parallel
    {
        std::vector<int64_t> localSumX(k, 0), localSumY(k, 0), localSumZ(k, 0);
        std::vector<int32_t> localCounts(k, 0);
#pragma omp for nowait
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            int32_t c = clusterAssignments[i];
            localSumX[c] += pc.x[i];
            localSumY[c] += pc.y[i];
            localSumZ[c] += pc.z[i];
            localCounts[c]++;
        }
#pragma omp critical
        {
            for (int32_t j = 0; j < k; ++j) {
                sumX[j] += localSumX[j];
                sumY[j] += localSumY[j];
                sumZ[j] += localSumZ[j];
                counts[j] += localCounts[j];
            }
        }
    }
    for (int32_t j = 0; j < k; ++j) {
        if (counts[j] > 0) {
            centroids[j].x = static_cast<int32_t>(sumX[j] / counts[j]);
            centroids[j].y = static_cast<int32_t>(sumY[j] / counts[j]);
            centroids[j].z = static_cast<int32_t>(sumZ[j] / counts[j]);
        }
    }
}

void updateCentroidsCPUFloat(const PointCloudFloat &pc, std::vector<CentroidFloat> &centroids,
                             const int32_t *clusterAssignments) {
    int32_t k = static_cast<int32_t>(centroids.size());
    std::vector<double> sumX(k, 0), sumY(k, 0), sumZ(k, 0);
    std::vector<int32_t> counts(k, 0);
#pragma omp parallel
    {
        std::vector<double> localSumX(k, 0), localSumY(k, 0), localSumZ(k, 0);
        std::vector<int32_t> localCounts(k, 0);
#pragma omp for nowait
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            int32_t c = clusterAssignments[i];
            localSumX[c] += pc.x[i];
            localSumY[c] += pc.y[i];
            localSumZ[c] += pc.z[i];
            localCounts[c]++;
        }
#pragma omp critical
        {
            for (int32_t j = 0; j < k; ++j) {
                sumX[j] += localSumX[j];
                sumY[j] += localSumY[j];
                sumZ[j] += localSumZ[j];
                counts[j] += localCounts[j];
            }
        }
    }
    for (int32_t j = 0; j < k; ++j) {
        if (counts[j] > 0) {
            centroids[j].x = static_cast<float>(sumX[j] / counts[j]);
            centroids[j].y = static_cast<float>(sumY[j] / counts[j]);
            centroids[j].z = static_cast<float>(sumZ[j] / counts[j]);
        }
    }
}

void computeClusterCDFsCPUInt(const PointCloudInt &pc, const int32_t *clusterAssignments, int32_t k,
                              std::vector<std::vector<int32_t>> &clusterCDFs,
                              std::vector<int32_t> &clusterCdfMins,
                              std::vector<int32_t> &clusterCounts) {
    std::vector<std::vector<int32_t>> histograms(k, std::vector<int32_t>(INTENSITY_LEVELS, 0));
#pragma omp parallel
    {
        std::vector<std::vector<int32_t>> localHist(k, std::vector<int32_t>(INTENSITY_LEVELS, 0));
        std::vector<int32_t> localCounts(k, 0);
#pragma omp for nowait
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            int32_t c = clusterAssignments[i];
            localHist[c][pc.intensity[i]]++;
            localCounts[c]++;
        }
#pragma omp critical
        {
            for (int32_t j = 0; j < k; ++j) {
                clusterCounts[j] += localCounts[j];
                for (int32_t v = 0; v < INTENSITY_LEVELS; ++v)
                    histograms[j][v] += localHist[j][v];
            }
        }
    }
    for (int32_t j = 0; j < k; ++j) {
        clusterCDFs[j][0] = histograms[j][0];
        for (int32_t v = 1; v < INTENSITY_LEVELS; ++v)
            clusterCDFs[j][v] = clusterCDFs[j][v - 1] + histograms[j][v];
        clusterCdfMins[j] = -1;
        for (int32_t v = 0; v < INTENSITY_LEVELS; ++v) {
            if (clusterCDFs[j][v] > 0) {
                clusterCdfMins[j] = clusterCDFs[j][v];
                break;
            }
        }
    }
}

void computeClusterCDFsCPUFloat(const PointCloudFloat &pc, const int32_t *clusterAssignments,
                                int32_t k, std::vector<std::vector<int32_t>> &clusterCDFs,
                                std::vector<int32_t> &clusterCdfMins,
                                std::vector<int32_t> &clusterCounts) {
    std::vector<std::vector<int32_t>> histograms(k, std::vector<int32_t>(INTENSITY_LEVELS, 0));
#pragma omp parallel
    {
        std::vector<std::vector<int32_t>> localHist(k, std::vector<int32_t>(INTENSITY_LEVELS, 0));
        std::vector<int32_t> localCounts(k, 0);
#pragma omp for nowait
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            int32_t c = clusterAssignments[i];
            localHist[c][pc.intensity[i]]++;
            localCounts[c]++;
        }
#pragma omp critical
        {
            for (int32_t j = 0; j < k; ++j) {
                clusterCounts[j] += localCounts[j];
                for (int32_t v = 0; v < INTENSITY_LEVELS; ++v)
                    histograms[j][v] += localHist[j][v];
            }
        }
    }
    for (int32_t j = 0; j < k; ++j) {
        clusterCDFs[j][0] = histograms[j][0];
        for (int32_t v = 1; v < INTENSITY_LEVELS; ++v)
            clusterCDFs[j][v] = clusterCDFs[j][v - 1] + histograms[j][v];
        clusterCdfMins[j] = -1;
        for (int32_t v = 0; v < INTENSITY_LEVELS; ++v) {
            if (clusterCDFs[j][v] > 0) {
                clusterCdfMins[j] = clusterCDFs[j][v];
                break;
            }
        }
    }
}

float kMeansCPUInt(const PointCloudInt &pc, int32_t k, int32_t iterations, int32_t *newIntensity) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<CentroidInt> centroids(k);
    for (int32_t j = 0; j < k; ++j)
        centroids[j] = {pc.x[j], pc.y[j], pc.z[j]};
    std::vector<int32_t> clusterAssignments(pc.numPoints);
    for (int32_t t = 0; t < iterations; ++t) {
        assignClustersCPUInt(pc, centroids, clusterAssignments.data());
        updateCentroidsCPUInt(pc, centroids, clusterAssignments.data());
    }
    std::vector<std::vector<int32_t>> clusterCDFs(k, std::vector<int32_t>(INTENSITY_LEVELS));
    std::vector<int32_t> clusterCdfMins(k);
    std::vector<int32_t> clusterCounts(k, 0);
    computeClusterCDFsCPUInt(pc, clusterAssignments.data(), k, clusterCDFs, clusterCdfMins,
                             clusterCounts);
#pragma omp parallel for
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        int32_t c = clusterAssignments[i];
        int32_t originalIntensity = pc.intensity[i];
        int32_t cdf = clusterCDFs[c][originalIntensity];
        int32_t cdfMin = clusterCdfMins[c];
        int32_t n = clusterCounts[c];
        newIntensity[i] = computeIntensityCPU(originalIntensity, cdf, cdfMin, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float kMeansCPUFloat(const PointCloudFloat &pc, int32_t k, int32_t iterations,
                     int32_t *newIntensity) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<CentroidFloat> centroids(k);
    for (int32_t j = 0; j < k; ++j)
        centroids[j] = {pc.x[j], pc.y[j], pc.z[j]};
    std::vector<int32_t> clusterAssignments(pc.numPoints);
    for (int32_t t = 0; t < iterations; ++t) {
        assignClustersCPUFloat(pc, centroids, clusterAssignments.data());
        updateCentroidsCPUFloat(pc, centroids, clusterAssignments.data());
    }
    std::vector<std::vector<int32_t>> clusterCDFs(k, std::vector<int32_t>(INTENSITY_LEVELS));
    std::vector<int32_t> clusterCdfMins(k);
    std::vector<int32_t> clusterCounts(k, 0);
    computeClusterCDFsCPUFloat(pc, clusterAssignments.data(), k, clusterCDFs, clusterCdfMins,
                               clusterCounts);
#pragma omp parallel for
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        int32_t c = clusterAssignments[i];
        int32_t originalIntensity = pc.intensity[i];
        int32_t cdf = clusterCDFs[c][originalIntensity];
        int32_t cdfMin = clusterCdfMins[c];
        int32_t n = clusterCounts[c];
        newIntensity[i] = computeIntensityCPU(originalIntensity, cdf, cdfMin, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

__global__ void assignClustersGPUInt(const int32_t *d_x, const int32_t *d_y, const int32_t *d_z,
                                     const int32_t *d_cx, const int32_t *d_cy, const int32_t *d_cz,
                                     int32_t *d_clusterAssignments, int32_t *d_changed,
                                     int32_t numPoints, int32_t k) {
    extern __shared__ int32_t s_centroids_int[];
    int32_t *s_cx = s_centroids_int;
    int32_t *s_cy = &s_centroids_int[k];
    int32_t *s_cz = &s_centroids_int[2 * k];

    for (int32_t j = threadIdx.x; j < k; j += blockDim.x) {
        s_cx[j] = d_cx[j];
        s_cy[j] = d_cy[j];
        s_cz[j] = d_cz[j];
    }
    __syncthreads();

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;

    int32_t px = d_x[i];
    int32_t py = d_y[i];
    int32_t pz = d_z[i];
    int32_t minDist = INT32_MAX;
    int32_t bestCluster = 0;

    for (int32_t j = 0; j < k; ++j) {
        int32_t dist = euclideanDistanceGPUInt(px, py, pz, s_cx[j], s_cy[j], s_cz[j]);
        bool closer = dist < minDist;
        if (!closer && dist == minDist) {
            if (s_cx[j] < s_cx[bestCluster])
                closer = true;
            else if (s_cx[j] == s_cx[bestCluster]) {
                if (s_cy[j] < s_cy[bestCluster])
                    closer = true;
                else if (s_cy[j] == s_cy[bestCluster] && s_cz[j] < s_cz[bestCluster])
                    closer = true;
            }
        }
        if (closer) {
            minDist = dist;
            bestCluster = j;
        }
    }
    int32_t oldCluster = d_clusterAssignments[i];
    if (oldCluster != bestCluster) {
        d_clusterAssignments[i] = bestCluster;
        *d_changed = 1;
    }
}

__global__ void assignClustersGPUFloat(const float *d_x, const float *d_y, const float *d_z,
                                       const float *d_cx, const float *d_cy, const float *d_cz,
                                       int32_t *d_clusterAssignments, int32_t *d_changed,
                                       int32_t numPoints, int32_t k) {
    extern __shared__ float s_centroids_float[];
    float *s_cx = s_centroids_float;
    float *s_cy = &s_centroids_float[k];
    float *s_cz = &s_centroids_float[2 * k];

    for (int32_t j = threadIdx.x; j < k; j += blockDim.x) {
        s_cx[j] = d_cx[j];
        s_cy[j] = d_cy[j];
        s_cz[j] = d_cz[j];
    }
    __syncthreads();

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;

    float px = d_x[i];
    float py = d_y[i];
    float pz = d_z[i];
    float minDist = FLT_MAX;
    int32_t bestCluster = 0;

    for (int32_t j = 0; j < k; ++j) {
        float dist = euclideanDistanceGPUFloat(px, py, pz, s_cx[j], s_cy[j], s_cz[j]);
        bool closer = dist < minDist;
        if (!closer && dist == minDist) {
            if (s_cx[j] < s_cx[bestCluster])
                closer = true;
            else if (s_cx[j] == s_cx[bestCluster]) {
                if (s_cy[j] < s_cy[bestCluster])
                    closer = true;
                else if (s_cy[j] == s_cy[bestCluster] && s_cz[j] < s_cz[bestCluster])
                    closer = true;
            }
        }
        if (closer) {
            minDist = dist;
            bestCluster = j;
        }
    }
    int32_t oldCluster = d_clusterAssignments[i];
    if (oldCluster != bestCluster) {
        d_clusterAssignments[i] = bestCluster;
        *d_changed = 1;
    }
}

__global__ void accumulateCentroidsGPUInt(const int32_t *d_x, const int32_t *d_y,
                                          const int32_t *d_z, const int32_t *d_clusterAssignments,
                                          int32_t *d_sumX, int32_t *d_sumY, int32_t *d_sumZ,
                                          int32_t *d_counts, int32_t numPoints, int32_t k) {
    extern __shared__ int32_t s_sums_int[];
    int32_t *s_sumX = s_sums_int;
    int32_t *s_sumY = &s_sums_int[k];
    int32_t *s_sumZ = &s_sums_int[2 * k];
    int32_t *s_counts = &s_sums_int[3 * k];
    for (int32_t j = threadIdx.x; j < k; j += blockDim.x) {
        s_sumX[j] = 0;
        s_sumY[j] = 0;
        s_sumZ[j] = 0;
        s_counts[j] = 0;
    }
    __syncthreads();
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        int32_t c = d_clusterAssignments[i];
        atomicAdd(&s_sumX[c], d_x[i]);
        atomicAdd(&s_sumY[c], d_y[i]);
        atomicAdd(&s_sumZ[c], d_z[i]);
        atomicAdd(&s_counts[c], 1);
    }
    __syncthreads();
    for (int32_t j = threadIdx.x; j < k; j += blockDim.x) {
        if (s_counts[j] > 0) {
            atomicAdd(&d_sumX[j], s_sumX[j]);
            atomicAdd(&d_sumY[j], s_sumY[j]);
            atomicAdd(&d_sumZ[j], s_sumZ[j]);
            atomicAdd(&d_counts[j], s_counts[j]);
        }
    }
}

__global__ void accumulateCentroidsGPUFloat(const float *d_x, const float *d_y, const float *d_z,
                                            const int32_t *d_clusterAssignments, float *d_sumX,
                                            float *d_sumY, float *d_sumZ, int32_t *d_counts,
                                            int32_t numPoints, int32_t k) {
    extern __shared__ float s_sums_float[];
    float *s_sumX = s_sums_float;
    float *s_sumY = &s_sums_float[k];
    float *s_sumZ = &s_sums_float[2 * k];
    int32_t *s_counts = reinterpret_cast<int32_t *>(&s_sums_float[3 * k]);
    for (int32_t j = threadIdx.x; j < k; j += blockDim.x) {
        s_sumX[j] = 0;
        s_sumY[j] = 0;
        s_sumZ[j] = 0;
        s_counts[j] = 0;
    }
    __syncthreads();
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        int32_t c = d_clusterAssignments[i];
        atomicAdd(&s_sumX[c], d_x[i]);
        atomicAdd(&s_sumY[c], d_y[i]);
        atomicAdd(&s_sumZ[c], d_z[i]);
        atomicAdd(&s_counts[c], 1);
    }
    __syncthreads();
    for (int32_t j = threadIdx.x; j < k; j += blockDim.x) {
        if (s_counts[j] > 0) {
            atomicAdd(&d_sumX[j], s_sumX[j]);
            atomicAdd(&d_sumY[j], s_sumY[j]);
            atomicAdd(&d_sumZ[j], s_sumZ[j]);
            atomicAdd(&d_counts[j], s_counts[j]);
        }
    }
}

__global__ void updateCentroidsGPUInt(int32_t *d_cx, int32_t *d_cy, int32_t *d_cz,
                                      const int32_t *d_sumX, const int32_t *d_sumY,
                                      const int32_t *d_sumZ, const int32_t *d_counts, int32_t k) {
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k)
        return;
    int32_t count = d_counts[j];
    if (count > 0) {
        d_cx[j] = d_sumX[j] / count;
        d_cy[j] = d_sumY[j] / count;
        d_cz[j] = d_sumZ[j] / count;
    }
}

__global__ void updateCentroidsGPUFloat(float *d_cx, float *d_cy, float *d_cz, const float *d_sumX,
                                        const float *d_sumY, const float *d_sumZ,
                                        const int32_t *d_counts, int32_t k) {
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k)
        return;
    int32_t count = d_counts[j];
    if (count > 0) {
        d_cx[j] = d_sumX[j] / count;
        d_cy[j] = d_sumY[j] / count;
        d_cz[j] = d_sumZ[j] / count;
    }
}

__global__ void computeHistogramsGPUInt(const int32_t *d_intensity,
                                        const int32_t *d_clusterAssignments, int32_t *d_histograms,
                                        int32_t numPoints, int32_t k) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        int32_t c = d_clusterAssignments[i];
        int32_t intensity = d_intensity[i];
        atomicAdd(&d_histograms[c * INTENSITY_LEVELS + intensity], 1);
    }
}

__global__ void computeHistogramsGPUFloat(const int32_t *d_intensity,
                                          const int32_t *d_clusterAssignments,
                                          int32_t *d_histograms, int32_t numPoints, int32_t k) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPoints) {
        int32_t c = d_clusterAssignments[i];
        int32_t intensity = d_intensity[i];
        atomicAdd(&d_histograms[c * INTENSITY_LEVELS + intensity], 1);
    }
}

__global__ void computeCDFsGPUInt(const int32_t *d_histograms, int32_t *d_cdf, int32_t *d_cdfMin,
                                  int32_t k) {
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k)
        return;
    int32_t offset = c * INTENSITY_LEVELS;
    int32_t cdf = 0;
    int32_t cdfMin = -1;
    for (int32_t v = 0; v < INTENSITY_LEVELS; ++v) {
        cdf += d_histograms[offset + v];
        d_cdf[offset + v] = cdf;
        if (cdfMin == -1 && cdf > 0)
            cdfMin = cdf;
    }
    d_cdfMin[c] = cdfMin;
}

__global__ void computeCDFsGPUFloat(const int32_t *d_histograms, int32_t *d_cdf, int32_t *d_cdfMin,
                                    int32_t k) {
    int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k)
        return;
    int32_t offset = c * INTENSITY_LEVELS;
    int32_t cdf = 0;
    int32_t cdfMin = -1;
    for (int32_t v = 0; v < INTENSITY_LEVELS; ++v) {
        cdf += d_histograms[offset + v];
        d_cdf[offset + v] = cdf;
        if (cdfMin == -1 && cdf > 0)
            cdfMin = cdf;
    }
    d_cdfMin[c] = cdfMin;
}

__global__ void remapIntensityGPUInt(const int32_t *d_intensity,
                                     const int32_t *d_clusterAssignments, const int32_t *d_cdf,
                                     const int32_t *d_cdfMin, const int32_t *d_counts,
                                     int32_t *d_newIntensity, int32_t numPoints) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    int32_t c = d_clusterAssignments[i];
    int32_t originalIntensity = d_intensity[i];
    int32_t cdf = d_cdf[c * INTENSITY_LEVELS + originalIntensity];
    int32_t cdfMin = d_cdfMin[c];
    int32_t n = d_counts[c];
    d_newIntensity[i] = computeIntensityGPU(originalIntensity, cdf, cdfMin, n);
}

__global__ void remapIntensityGPUFloat(const int32_t *d_intensity,
                                       const int32_t *d_clusterAssignments, const int32_t *d_cdf,
                                       const int32_t *d_cdfMin, const int32_t *d_counts,
                                       int32_t *d_newIntensity, int32_t numPoints) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    int32_t c = d_clusterAssignments[i];
    int32_t originalIntensity = d_intensity[i];
    int32_t cdf = d_cdf[c * INTENSITY_LEVELS + originalIntensity];
    int32_t cdfMin = d_cdfMin[c];
    int32_t n = d_counts[c];
    d_newIntensity[i] = computeIntensityGPU(originalIntensity, cdf, cdfMin, n);
}

float kMeansGPUInt(const PointCloudInt &h_pc, int32_t k, int32_t iterations,
                   int32_t *h_newIntensity) {
    int32_t numPoints = h_pc.numPoints;
    size_t intBytes = numPoints * sizeof(int32_t);
    size_t kIntBytes = k * sizeof(int32_t);
    int32_t *d_x, *d_y, *d_z, *d_intensity, *d_newIntensity, *d_clusterAssignments, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_x, intBytes));
    CUDA_CHECK(cudaMalloc(&d_y, intBytes));
    CUDA_CHECK(cudaMalloc(&d_z, intBytes));
    CUDA_CHECK(cudaMalloc(&d_intensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_newIntensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_clusterAssignments, intBytes));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int32_t)));
    int32_t *d_cx, *d_cy, *d_cz, *d_sumX, *d_sumY, *d_sumZ, *d_counts;
    CUDA_CHECK(cudaMalloc(&d_cx, kIntBytes));
    CUDA_CHECK(cudaMalloc(&d_cy, kIntBytes));
    CUDA_CHECK(cudaMalloc(&d_cz, kIntBytes));
    CUDA_CHECK(cudaMalloc(&d_sumX, kIntBytes));
    CUDA_CHECK(cudaMalloc(&d_sumY, kIntBytes));
    CUDA_CHECK(cudaMalloc(&d_sumZ, kIntBytes));
    CUDA_CHECK(cudaMalloc(&d_counts, kIntBytes));
    size_t histBytes = k * INTENSITY_LEVELS * sizeof(int32_t);
    int32_t *d_histograms, *d_cdf, *d_cdfMin;
    CUDA_CHECK(cudaMalloc(&d_histograms, histBytes));
    CUDA_CHECK(cudaMalloc(&d_cdf, histBytes));
    CUDA_CHECK(cudaMalloc(&d_cdfMin, kIntBytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_pc.x, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_pc.y, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_pc.z, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensity, h_pc.intensity, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cx, h_pc.x, kIntBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, h_pc.y, kIntBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cz, h_pc.z, kIntBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int32_t blockSize = 256;
    int32_t gridSizePoints = (numPoints + blockSize - 1) / blockSize;
    int32_t gridSizeK = (k + blockSize - 1) / blockSize;
    size_t assignSharedMem = 3 * k * sizeof(int32_t);
    size_t accumSharedMem = 4 * k * sizeof(int32_t);
    CUDA_CHECK(cudaMemset(d_clusterAssignments, -1, intBytes));
    for (int32_t t = 0; t < iterations; ++t) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int32_t)));
        assignClustersGPUInt<<<gridSizePoints, blockSize, assignSharedMem>>>(
            d_x, d_y, d_z, d_cx, d_cy, d_cz, d_clusterAssignments, d_changed, numPoints, k);
        CUDA_CHECK_KERNEL();

        int32_t h_changed = 0;
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (h_changed == 0)
            break;

        CUDA_CHECK(cudaMemset(d_sumX, 0, kIntBytes));
        CUDA_CHECK(cudaMemset(d_sumY, 0, kIntBytes));
        CUDA_CHECK(cudaMemset(d_sumZ, 0, kIntBytes));
        CUDA_CHECK(cudaMemset(d_counts, 0, kIntBytes));
        accumulateCentroidsGPUInt<<<gridSizePoints, blockSize, accumSharedMem>>>(
            d_x, d_y, d_z, d_clusterAssignments, d_sumX, d_sumY, d_sumZ, d_counts, numPoints, k);
        CUDA_CHECK_KERNEL();
        updateCentroidsGPUInt<<<gridSizeK, blockSize>>>(d_cx, d_cy, d_cz, d_sumX, d_sumY, d_sumZ,
                                                        d_counts, k);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaMemset(d_histograms, 0, histBytes));
    computeHistogramsGPUInt<<<gridSizePoints, blockSize>>>(d_intensity, d_clusterAssignments,
                                                           d_histograms, numPoints, k);
    CUDA_CHECK_KERNEL();
    computeCDFsGPUInt<<<gridSizeK, blockSize>>>(d_histograms, d_cdf, d_cdfMin, k);
    CUDA_CHECK_KERNEL();
    remapIntensityGPUInt<<<gridSizePoints, blockSize>>>(
        d_intensity, d_clusterAssignments, d_cdf, d_cdfMin, d_counts, d_newIntensity, numPoints);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_newIntensity, d_newIntensity, intBytes, cudaMemcpyDeviceToHost));
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_intensity);
    cudaFree(d_newIntensity);
    cudaFree(d_clusterAssignments);
    cudaFree(d_changed);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_cz);
    cudaFree(d_sumX);
    cudaFree(d_sumY);
    cudaFree(d_sumZ);
    cudaFree(d_counts);
    cudaFree(d_histograms);
    cudaFree(d_cdf);
    cudaFree(d_cdfMin);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

float kMeansGPUFloat(const PointCloudFloat &h_pc, int32_t k, int32_t iterations,
                     int32_t *h_newIntensity) {
    int32_t numPoints = h_pc.numPoints;
    size_t fBytes = numPoints * sizeof(float);
    size_t kfBytes = k * sizeof(float);
    size_t intBytes = numPoints * sizeof(int32_t);
    size_t kIntBytes = k * sizeof(int32_t);
    float *d_x, *d_y, *d_z, *d_cx, *d_cy, *d_cz, *d_sumX, *d_sumY, *d_sumZ;
    int32_t *d_intensity, *d_newIntensity, *d_clusterAssignments, *d_counts, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_x, fBytes));
    CUDA_CHECK(cudaMalloc(&d_y, fBytes));
    CUDA_CHECK(cudaMalloc(&d_z, fBytes));
    CUDA_CHECK(cudaMalloc(&d_intensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_newIntensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_clusterAssignments, intBytes));
    CUDA_CHECK(cudaMalloc(&d_cx, kfBytes));
    CUDA_CHECK(cudaMalloc(&d_cy, kfBytes));
    CUDA_CHECK(cudaMalloc(&d_cz, kfBytes));
    CUDA_CHECK(cudaMalloc(&d_sumX, kfBytes));
    CUDA_CHECK(cudaMalloc(&d_sumY, kfBytes));
    CUDA_CHECK(cudaMalloc(&d_sumZ, kfBytes));
    CUDA_CHECK(cudaMalloc(&d_counts, kIntBytes));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int32_t)));
    size_t histBytes = k * INTENSITY_LEVELS * sizeof(int32_t);
    int32_t *d_histograms, *d_cdf, *d_cdfMin;
    CUDA_CHECK(cudaMalloc(&d_histograms, histBytes));
    CUDA_CHECK(cudaMalloc(&d_cdf, histBytes));
    CUDA_CHECK(cudaMalloc(&d_cdfMin, kIntBytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_pc.x, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_pc.y, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_pc.z, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensity, h_pc.intensity, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cx, h_pc.x, kfBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, h_pc.y, kfBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cz, h_pc.z, kfBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int32_t blockSize = 256;
    int32_t gridSizePoints = (numPoints + blockSize - 1) / blockSize;
    int32_t gridSizeK = (k + blockSize - 1) / blockSize;
    size_t assignSharedMem = 3 * k * sizeof(float);
    size_t accumSharedMem = 3 * k * sizeof(float) + k * sizeof(int32_t);
    CUDA_CHECK(cudaMemset(d_clusterAssignments, -1, intBytes));
    for (int32_t t = 0; t < iterations; ++t) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int32_t)));
        assignClustersGPUFloat<<<gridSizePoints, blockSize, assignSharedMem>>>(
            d_x, d_y, d_z, d_cx, d_cy, d_cz, d_clusterAssignments, d_changed, numPoints, k);
        CUDA_CHECK_KERNEL();

        int32_t h_changed = 0;
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (h_changed == 0)
            break;

        CUDA_CHECK(cudaMemset(d_sumX, 0, kfBytes));
        CUDA_CHECK(cudaMemset(d_sumY, 0, kfBytes));
        CUDA_CHECK(cudaMemset(d_sumZ, 0, kfBytes));
        CUDA_CHECK(cudaMemset(d_counts, 0, kIntBytes));
        accumulateCentroidsGPUFloat<<<gridSizePoints, blockSize, accumSharedMem>>>(
            d_x, d_y, d_z, d_clusterAssignments, d_sumX, d_sumY, d_sumZ, d_counts, numPoints, k);
        CUDA_CHECK_KERNEL();
        updateCentroidsGPUFloat<<<gridSizeK, blockSize>>>(d_cx, d_cy, d_cz, d_sumX, d_sumY, d_sumZ,
                                                          d_counts, k);
        CUDA_CHECK_KERNEL();
    }
    CUDA_CHECK(cudaMemset(d_histograms, 0, histBytes));
    computeHistogramsGPUFloat<<<gridSizePoints, blockSize>>>(d_intensity, d_clusterAssignments,
                                                             d_histograms, numPoints, k);
    CUDA_CHECK_KERNEL();
    computeCDFsGPUFloat<<<gridSizeK, blockSize>>>(d_histograms, d_cdf, d_cdfMin, k);
    CUDA_CHECK_KERNEL();
    remapIntensityGPUFloat<<<gridSizePoints, blockSize>>>(
        d_intensity, d_clusterAssignments, d_cdf, d_cdfMin, d_counts, d_newIntensity, numPoints);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_newIntensity, d_newIntensity, intBytes, cudaMemcpyDeviceToHost));
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_intensity);
    cudaFree(d_newIntensity);
    cudaFree(d_clusterAssignments);
    cudaFree(d_changed);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_cz);
    cudaFree(d_sumX);
    cudaFree(d_sumY);
    cudaFree(d_sumZ);
    cudaFree(d_counts);
    cudaFree(d_histograms);
    cudaFree(d_cdf);
    cudaFree(d_cdfMin);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

} // namespace kMeans