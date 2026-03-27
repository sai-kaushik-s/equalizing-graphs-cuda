#include "knn.cuh"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>

namespace knn {

struct NeighborDistanceInt {
    int32_t distance;
    int32_t x, y, z;
    int32_t pointIdx;
    bool operator<(const NeighborDistanceInt &other) const {
        if (distance != other.distance)
            return distance < other.distance;
        if (x != other.x)
            return x < other.x;
        if (y != other.y)
            return y < other.y;
        return z < other.z;
    }
};

struct NeighborDistanceFloat {
    float distance;
    float x, y, z;
    int32_t pointIdx;
    bool operator<(const NeighborDistanceFloat &other) const {
        if (distance != other.distance)
            return distance < other.distance;
        if (x != other.x)
            return x < other.x;
        if (y != other.y)
            return y < other.y;
        return z < other.z;
    }
};

void findKNNCPUInt(const PointCloudInt &pc, int32_t queryIdx, int32_t k,
                   std::vector<NeighborDistanceInt> &neighborDistances) {
    int32_t queryX = pc.x[queryIdx];
    int32_t queryY = pc.y[queryIdx];
    int32_t queryZ = pc.z[queryIdx];
    int32_t count = 0;
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        if (i == queryIdx)
            continue;
        int32_t sqDist = euclideanDistanceCPUInt(queryX, queryY, queryZ, pc.x[i], pc.y[i], pc.z[i]);
        neighborDistances[count++] = {sqDist, pc.x[i], pc.y[i], pc.z[i], i};
    }
    std::nth_element(neighborDistances.begin(), neighborDistances.begin() + k,
                     neighborDistances.end());
}

void findKNNCPUFloat(const PointCloudFloat &pc, int32_t queryIdx, int32_t k,
                     std::vector<NeighborDistanceFloat> &neighborDistances) {
    float queryX = pc.x[queryIdx];
    float queryY = pc.y[queryIdx];
    float queryZ = pc.z[queryIdx];
    int32_t count = 0;
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        if (i == queryIdx)
            continue;
        float sqDist = euclideanDistanceCPUFloat(queryX, queryY, queryZ, pc.x[i], pc.y[i], pc.z[i]);
        neighborDistances[count++] = {sqDist, pc.x[i], pc.y[i], pc.z[i], i};
    }
    std::nth_element(neighborDistances.begin(), neighborDistances.begin() + k,
                     neighborDistances.end());
}

int32_t computeCDFCPUInt(const PointCloudInt &pc, const std::vector<NeighborDistanceInt> &neighbors,
                         int32_t k, int32_t centerIntensity, int32_t *cdfLocal) {
    int32_t histLocal[INTENSITY_LEVELS] = {0};
    histLocal[centerIntensity]++;
    for (int32_t i = 0; i < k; ++i) {
        int32_t neighborIdx = neighbors[i].pointIdx;
        histLocal[pc.intensity[neighborIdx]]++;
    }
    cdfLocal[0] = histLocal[0];
    for (int32_t v = 1; v < INTENSITY_LEVELS; ++v)
        cdfLocal[v] = cdfLocal[v - 1] + histLocal[v];
    int32_t cdfMin = -1;
    for (int32_t v = 0; v < INTENSITY_LEVELS; ++v) {
        if (cdfLocal[v] > 0) {
            cdfMin = cdfLocal[v];
            break;
        }
    }
    return cdfMin;
}

int32_t computeCDFCPUFloat(const PointCloudFloat &pc,
                           const std::vector<NeighborDistanceFloat> &neighbors, int32_t k,
                           int32_t centerIntensity, int32_t *cdfLocal) {
    int32_t histLocal[INTENSITY_LEVELS] = {0};
    histLocal[centerIntensity]++;
    for (int32_t i = 0; i < k; ++i) {
        int32_t neighborIdx = neighbors[i].pointIdx;
        histLocal[pc.intensity[neighborIdx]]++;
    }
    cdfLocal[0] = histLocal[0];
    for (int32_t v = 1; v < INTENSITY_LEVELS; ++v)
        cdfLocal[v] = cdfLocal[v - 1] + histLocal[v];
    int32_t cdfMin = -1;
    for (int32_t v = 0; v < INTENSITY_LEVELS; ++v) {
        if (cdfLocal[v] > 0) {
            cdfMin = cdfLocal[v];
            break;
        }
    }
    return cdfMin;
}

float knnCPUInt(const PointCloudInt &pc, int32_t k, int32_t *newIntensity) {
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
        std::vector<NeighborDistanceInt> distancesLocal(pc.numPoints);
        int32_t cdfLocal[INTENSITY_LEVELS];
#pragma omp for
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            findKNNCPUInt(pc, i, k, distancesLocal);
            int32_t intensity = pc.intensity[i];
            int32_t cdfMin = computeCDFCPUInt(pc, distancesLocal, k, intensity, cdfLocal);
            int32_t cdf = cdfLocal[intensity];
            newIntensity[i] = computeIntensityCPU(intensity, cdf, cdfMin, k + 1);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float knnCPUFloat(const PointCloudFloat &pc, int32_t k, int32_t *newIntensity) {
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
        std::vector<NeighborDistanceFloat> distancesLocal(pc.numPoints);
        int32_t cdfLocal[INTENSITY_LEVELS];
#pragma omp for
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            findKNNCPUFloat(pc, i, k, distancesLocal);
            int32_t intensity = pc.intensity[i];
            int32_t cdfMin = computeCDFCPUFloat(pc, distancesLocal, k, intensity, cdfLocal);
            int32_t cdf = cdfLocal[intensity];
            newIntensity[i] = computeIntensityCPU(intensity, cdf, cdfMin, k + 1);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

__device__ void findKNNGPUInt(const int32_t *d_x, const int32_t *d_y, const int32_t *d_z,
                              const int32_t *d_intensity, int32_t numPoints, int32_t queryIdx,
                              int32_t queryX, int32_t queryY, int32_t queryZ, int32_t k,
                              int32_t *distancesLocal, int32_t *intensitiesLocal, int32_t *nx,
                              int32_t *ny, int32_t *nz) {
    for (int32_t i = 0; i < k; ++i) {
        distancesLocal[i] = INT32_MAX;
        intensitiesLocal[i] = -1;
        nx[i] = INT32_MAX;
        ny[i] = INT32_MAX;
        nz[i] = INT32_MAX;
    }
    for (int32_t i = 0; i < numPoints; ++i) {
        if (i == queryIdx)
            continue;
        int32_t sqDist = euclideanDistanceGPUInt(queryX, queryY, queryZ, d_x[i], d_y[i], d_z[i]);
        int32_t px = d_x[i];
        int32_t py = d_y[i];
        int32_t pz = d_z[i];

        bool closer = sqDist < distancesLocal[k - 1];
        if (!closer && sqDist == distancesLocal[k - 1]) {
            if (px < nx[k - 1])
                closer = true;
            else if (px == nx[k - 1]) {
                if (py < ny[k - 1])
                    closer = true;
                else if (py == ny[k - 1] && pz < nz[k - 1])
                    closer = true;
            }
        }

        if (closer) {
            int32_t pos = k - 1;
            while (pos > 0) {
                bool move = sqDist < distancesLocal[pos - 1];
                if (!move && sqDist == distancesLocal[pos - 1]) {
                    if (px < nx[pos - 1])
                        move = true;
                    else if (px == nx[pos - 1]) {
                        if (py < ny[pos - 1])
                            move = true;
                        else if (py == ny[pos - 1] && pz < nz[pos - 1])
                            move = true;
                    }
                }
                if (!move)
                    break;

                distancesLocal[pos] = distancesLocal[pos - 1];
                intensitiesLocal[pos] = intensitiesLocal[pos - 1];
                nx[pos] = nx[pos - 1];
                ny[pos] = ny[pos - 1];
                nz[pos] = nz[pos - 1];
                pos--;
            }
            distancesLocal[pos] = sqDist;
            intensitiesLocal[pos] = d_intensity[i];
            nx[pos] = px;
            ny[pos] = py;
            nz[pos] = pz;
        }
    }
}

__device__ void findKNNGPUFloat(const float *d_x, const float *d_y, const float *d_z,
                                const int32_t *d_intensity, int32_t numPoints, int32_t queryIdx,
                                float queryX, float queryY, float queryZ, int32_t k,
                                float *distancesLocal, int32_t *intensitiesLocal, float *nx,
                                float *ny, float *nz) {
    for (int32_t i = 0; i < k; ++i) {
        distancesLocal[i] = FLT_MAX;
        intensitiesLocal[i] = -1;
        nx[i] = FLT_MAX;
        ny[i] = FLT_MAX;
        nz[i] = FLT_MAX;
    }
    for (int32_t i = 0; i < numPoints; ++i) {
        if (i == queryIdx)
            continue;
        float sqDist = euclideanDistanceGPUFloat(queryX, queryY, queryZ, d_x[i], d_y[i], d_z[i]);
        float px = d_x[i];
        float py = d_y[i];
        float pz = d_z[i];

        bool closer = sqDist < distancesLocal[k - 1];
        if (!closer && sqDist == distancesLocal[k - 1]) {
            if (px < nx[k - 1])
                closer = true;
            else if (px == nx[k - 1]) {
                if (py < ny[k - 1])
                    closer = true;
                else if (py == ny[k - 1] && pz < nz[k - 1])
                    closer = true;
            }
        }

        if (closer) {
            int32_t pos = k - 1;
            while (pos > 0) {
                bool move = sqDist < distancesLocal[pos - 1];
                if (!move && sqDist == distancesLocal[pos - 1]) {
                    if (px < nx[pos - 1])
                        move = true;
                    else if (px == nx[pos - 1]) {
                        if (py < ny[pos - 1])
                            move = true;
                        else if (py == ny[pos - 1] && pz < nz[pos - 1])
                            move = true;
                    }
                }
                if (!move)
                    break;

                distancesLocal[pos] = distancesLocal[pos - 1];
                intensitiesLocal[pos] = intensitiesLocal[pos - 1];
                nx[pos] = nx[pos - 1];
                ny[pos] = ny[pos - 1];
                nz[pos] = nz[pos - 1];
                pos--;
            }
            distancesLocal[pos] = sqDist;
            intensitiesLocal[pos] = d_intensity[i];
            nx[pos] = px;
            ny[pos] = py;
            nz[pos] = pz;
        }
    }
}

__device__ int32_t computeCDFGPU(const int32_t *intensitiesLocal, int32_t k,
                                 int32_t centerIntensity, int32_t *cdfLocal) {
    int32_t histLocal[INTENSITY_LEVELS] = {0};
    histLocal[centerIntensity]++;
    for (int32_t i = 0; i < k; ++i) {
        if (intensitiesLocal[i] != -1)
            histLocal[intensitiesLocal[i]]++;
    }
    cdfLocal[0] = histLocal[0];
    for (int32_t v = 1; v < INTENSITY_LEVELS; ++v)
        cdfLocal[v] = cdfLocal[v - 1] + histLocal[v];
    int32_t cdfMin = -1;
    for (int32_t v = 0; v < INTENSITY_LEVELS; ++v) {
        if (cdfLocal[v] > 0) {
            cdfMin = cdfLocal[v];
            break;
        }
    }
    return cdfMin;
}

__global__ void knnKernelInt(const int32_t *d_x, const int32_t *d_y, const int32_t *d_z,
                             const int32_t *d_intensity, int32_t *d_newIntensity, int32_t numPoints,
                             int32_t k) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    int32_t distancesLocal[MAX_K];
    int32_t intensitiesLocal[MAX_K];
    int32_t nx[MAX_K], ny[MAX_K], nz[MAX_K];
    findKNNGPUInt(d_x, d_y, d_z, d_intensity, numPoints, i, d_x[i], d_y[i], d_z[i], k,
                  distancesLocal, intensitiesLocal, nx, ny, nz);
    int32_t intensity = d_intensity[i];
    int32_t cdfLocal[INTENSITY_LEVELS];
    int32_t cdfMin = computeCDFGPU(intensitiesLocal, k, intensity, cdfLocal);
    int32_t cdf = cdfLocal[intensity];
    d_newIntensity[i] = computeIntensityGPU(intensity, cdf, cdfMin, k + 1);
}

__global__ void knnKernelFloat(const float *d_x, const float *d_y, const float *d_z,
                               const int32_t *d_intensity, int32_t *d_newIntensity,
                               int32_t numPoints, int32_t k) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    float distancesLocal[MAX_K];
    int32_t intensitiesLocal[MAX_K];
    float nx[MAX_K], ny[MAX_K], nz[MAX_K];
    findKNNGPUFloat(d_x, d_y, d_z, d_intensity, numPoints, i, d_x[i], d_y[i], d_z[i], k,
                    distancesLocal, intensitiesLocal, nx, ny, nz);
    int32_t intensity = d_intensity[i];
    int32_t cdfLocal[INTENSITY_LEVELS];
    int32_t cdfMin = computeCDFGPU(intensitiesLocal, k, intensity, cdfLocal);
    int32_t cdf = cdfLocal[intensity];
    d_newIntensity[i] = computeIntensityGPU(intensity, cdf, cdfMin, k + 1);
}

float knnGPUInt(const PointCloudInt &h_pc, int32_t k, int32_t *h_newIntensity) {
    int32_t numPoints = h_pc.numPoints;
    size_t intBytes = numPoints * sizeof(int32_t);
    int32_t *d_x, *d_y, *d_z, *d_intensity, *d_newIntensity;
    CUDA_CHECK(cudaMalloc(&d_x, intBytes));
    CUDA_CHECK(cudaMalloc(&d_y, intBytes));
    CUDA_CHECK(cudaMalloc(&d_z, intBytes));
    CUDA_CHECK(cudaMalloc(&d_intensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_newIntensity, intBytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_pc.x, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_pc.y, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_pc.z, intBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensity, h_pc.intensity, intBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int32_t blockSize = 256;
    int32_t gridSize = (numPoints + blockSize - 1) / blockSize;
    knnKernelInt<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_intensity, d_newIntensity, numPoints, k);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_newIntensity, d_newIntensity, intBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_intensity));
    CUDA_CHECK(cudaFree(d_newIntensity));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

float knnGPUFloat(const PointCloudFloat &h_pc, int32_t k, int32_t *h_newIntensity) {
    int32_t numPoints = h_pc.numPoints;
    size_t fBytes = numPoints * sizeof(float);
    size_t intBytes = numPoints * sizeof(int32_t);
    float *d_x, *d_y, *d_z;
    int32_t *d_intensity, *d_newIntensity;
    CUDA_CHECK(cudaMalloc(&d_x, fBytes));
    CUDA_CHECK(cudaMalloc(&d_y, fBytes));
    CUDA_CHECK(cudaMalloc(&d_z, fBytes));
    CUDA_CHECK(cudaMalloc(&d_intensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_newIntensity, intBytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_pc.x, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_pc.y, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_pc.z, fBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensity, h_pc.intensity, intBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int32_t blockSize = 256;
    int32_t gridSize = (numPoints + blockSize - 1) / blockSize;
    knnKernelFloat<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_intensity, d_newIntensity, numPoints,
                                            k);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_newIntensity, d_newIntensity, intBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_intensity));
    CUDA_CHECK(cudaFree(d_newIntensity));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

} // namespace knn