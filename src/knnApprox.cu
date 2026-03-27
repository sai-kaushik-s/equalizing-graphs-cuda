#include "knnApprox.cuh"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>
#include <unordered_map>
#include <vector>

#define HASH_TABLE_SIZE 1999993
#define MAX_RADIUS 50

namespace knnApprox {

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

template <typename T>
float calculateOptimalVoxelSizeImpl(const T *x, const T *y, const T *z, int32_t n) {
    float minX = FLT_MAX, maxX = -FLT_MAX;
    float minY = FLT_MAX, maxY = -FLT_MAX;
    float minZ = FLT_MAX, maxZ = -FLT_MAX;

    for (int32_t i = 0; i < n; ++i) {
        float fx = static_cast<float>(x[i]);
        float fy = static_cast<float>(y[i]);
        float fz = static_cast<float>(z[i]);

        if (fx < minX)
            minX = fx;
        if (fx > maxX)
            maxX = fx;
        if (fy < minY)
            minY = fy;
        if (fy > maxY)
            maxY = fy;
        if (fz < minZ)
            minZ = fz;
        if (fz > maxZ)
            maxZ = fz;
    }

    float volume = (maxX - minX) * (maxY - minY) * (maxZ - minZ);
    if (volume <= 0)
        return 1.0f;

    float density = static_cast<float>(n) / volume;
    return std::cbrt(16.0f / (27.0f * density));
}

float calculateOptimalVoxelSize(const PointCloudInt &pc) {
    return calculateOptimalVoxelSizeImpl(pc.x, pc.y, pc.z, pc.numPoints);
}

float calculateOptimalVoxelSize(const PointCloudFloat &pc) {
    return calculateOptimalVoxelSizeImpl(pc.x, pc.y, pc.z, pc.numPoints);
}

inline int64_t getVoxelHash(int32_t gx, int32_t gy, int32_t gz) {
    return (static_cast<int64_t>(gx) * 73856093) ^ (static_cast<int64_t>(gy) * 19349663) ^
           (static_cast<int64_t>(gz) * 83492791);
}

int32_t computeCDFCPUInt(const PointCloudInt &pc, const std::vector<NeighborDistanceInt> &neighbors,
                         int32_t actualK, int32_t centerIntensity, int32_t *cdfLocal) {
    int32_t histLocal[INTENSITY_LEVELS] = {0};
    histLocal[centerIntensity]++;
    for (int32_t i = 0; i < actualK; ++i) {
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
                           const std::vector<NeighborDistanceFloat> &neighbors, int32_t actualK,
                           int32_t centerIntensity, int32_t *cdfLocal) {
    int32_t histLocal[INTENSITY_LEVELS] = {0};
    histLocal[centerIntensity]++;
    for (int32_t i = 0; i < actualK; ++i) {
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

float knnApproxCPUInt(const PointCloudInt &pc, int32_t k, int32_t *newIntensity) {
    auto start = std::chrono::high_resolution_clock::now();
    float voxelSize = calculateOptimalVoxelSize(pc);
    std::unordered_map<int64_t, std::vector<int32_t>> voxelGrid;
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        int32_t gx = static_cast<int32_t>(pc.x[i] / voxelSize);
        int32_t gy = static_cast<int32_t>(pc.y[i] / voxelSize);
        int32_t gz = static_cast<int32_t>(pc.z[i] / voxelSize);
        int64_t hash = getVoxelHash(gx, gy, gz);
        voxelGrid[hash].push_back(i);
    }
#pragma omp parallel
    {
        std::vector<NeighborDistanceInt> localCandidates;
        int32_t cdfLocal[INTENSITY_LEVELS];
#pragma omp for
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            int32_t queryX = pc.x[i];
            int32_t queryY = pc.y[i];
            int32_t queryZ = pc.z[i];
            int32_t gx = static_cast<int32_t>(queryX / voxelSize);
            int32_t gy = static_cast<int32_t>(queryY / voxelSize);
            int32_t gz = static_cast<int32_t>(queryZ / voxelSize);

            localCandidates.clear();
            int32_t stopRadius = MAX_RADIUS;
            for (int32_t radius = 1; radius <= stopRadius; ++radius) {
                for (int32_t dx = -radius; dx <= radius; ++dx) {
                    for (int32_t dy = -radius; dy <= radius; ++dy) {
                        for (int32_t dz = -radius; dz <= radius; ++dz) {
                            if (std::abs(dx) < radius && std::abs(dy) < radius &&
                                std::abs(dz) < radius)
                                continue;
                            int64_t hash = getVoxelHash(gx + dx, gy + dy, gz + dz);
                            auto it = voxelGrid.find(hash);
                            if (it != voxelGrid.end()) {
                                for (int32_t candidateIdx : it->second) {
                                    if (candidateIdx == i)
                                        continue;
                                    int32_t sqDist = euclideanDistanceCPUInt(
                                        queryX, queryY, queryZ, pc.x[candidateIdx],
                                        pc.y[candidateIdx], pc.z[candidateIdx]);
                                    localCandidates.push_back({sqDist, pc.x[candidateIdx],
                                                               pc.y[candidateIdx],
                                                               pc.z[candidateIdx], candidateIdx});
                                }
                            }
                        }
                    }
                }
                if (static_cast<int32_t>(localCandidates.size()) >= k && stopRadius == MAX_RADIUS)
                    stopRadius = std::min(radius * 2, (int32_t)MAX_RADIUS);
            }
            int32_t actualK = std::min(k, static_cast<int32_t>(localCandidates.size()));
            if (actualK > 0) {
                std::nth_element(localCandidates.begin(), localCandidates.begin() + actualK,
                                 localCandidates.end());
                int32_t intensity = pc.intensity[i];
                int32_t cdfMin =
                    computeCDFCPUInt(pc, localCandidates, actualK, intensity, cdfLocal);
                int32_t cdf = cdfLocal[intensity];
                newIntensity[i] = computeIntensityCPU(intensity, cdf, cdfMin, k + 1);
            } else {
                newIntensity[i] = pc.intensity[i];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float knnApproxCPUFloat(const PointCloudFloat &pc, int32_t k, int32_t *newIntensity) {
    auto start = std::chrono::high_resolution_clock::now();
    float voxelSize = calculateOptimalVoxelSize(pc);
    std::unordered_map<int64_t, std::vector<int32_t>> voxelGrid;
    for (int32_t i = 0; i < pc.numPoints; ++i) {
        int32_t gx = static_cast<int32_t>(pc.x[i] / voxelSize);
        int32_t gy = static_cast<int32_t>(pc.y[i] / voxelSize);
        int32_t gz = static_cast<int32_t>(pc.z[i] / voxelSize);
        int64_t hash = getVoxelHash(gx, gy, gz);
        voxelGrid[hash].push_back(i);
    }
#pragma omp parallel
    {
        std::vector<NeighborDistanceFloat> localCandidates;
        int32_t cdfLocal[INTENSITY_LEVELS];
#pragma omp for
        for (int32_t i = 0; i < pc.numPoints; ++i) {
            float queryX = pc.x[i];
            float queryY = pc.y[i];
            float queryZ = pc.z[i];
            int32_t gx = static_cast<int32_t>(queryX / voxelSize);
            int32_t gy = static_cast<int32_t>(queryY / voxelSize);
            int32_t gz = static_cast<int32_t>(queryZ / voxelSize);

            localCandidates.clear();
            int32_t stopRadius = MAX_RADIUS;
            for (int32_t radius = 1; radius <= stopRadius; ++radius) {
                for (int32_t dx = -radius; dx <= radius; ++dx) {
                    for (int32_t dy = -radius; dy <= radius; ++dy) {
                        for (int32_t dz = -radius; dz <= radius; ++dz) {
                            if (std::abs(dx) < radius && std::abs(dy) < radius &&
                                std::abs(dz) < radius)
                                continue;
                            int64_t hash = getVoxelHash(gx + dx, gy + dy, gz + dz);
                            auto it = voxelGrid.find(hash);
                            if (it != voxelGrid.end()) {
                                for (int32_t candidateIdx : it->second) {
                                    if (candidateIdx == i)
                                        continue;
                                    float sqDist = euclideanDistanceCPUFloat(
                                        queryX, queryY, queryZ, pc.x[candidateIdx],
                                        pc.y[candidateIdx], pc.z[candidateIdx]);
                                    localCandidates.push_back({sqDist, pc.x[candidateIdx],
                                                               pc.y[candidateIdx],
                                                               pc.z[candidateIdx], candidateIdx});
                                }
                            }
                        }
                    }
                }
                if (static_cast<int32_t>(localCandidates.size()) >= k && stopRadius == MAX_RADIUS)
                    stopRadius = std::min(radius * 2, (int32_t)MAX_RADIUS);
            }
            int32_t actualK = std::min(k, static_cast<int32_t>(localCandidates.size()));
            if (actualK > 0) {
                std::nth_element(localCandidates.begin(), localCandidates.begin() + actualK,
                                 localCandidates.end());
                int32_t intensity = pc.intensity[i];
                int32_t cdfMin =
                    computeCDFCPUFloat(pc, localCandidates, actualK, intensity, cdfLocal);
                int32_t cdf = cdfLocal[intensity];
                newIntensity[i] = computeIntensityCPU(intensity, cdf, cdfMin, k + 1);
            } else {
                newIntensity[i] = pc.intensity[i];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

__device__ inline uint32_t getVoxelHashGPU(int32_t gx, int32_t gy, int32_t gz) {
    int64_t h = (static_cast<int64_t>(gx) * 73856093) ^ (static_cast<int64_t>(gy) * 19349663) ^
                (static_cast<int64_t>(gz) * 83492791);
    return static_cast<uint32_t>(h);
}

__device__ int32_t findApproxKNNGPUInt(const int32_t *d_x, const int32_t *d_y, const int32_t *d_z,
                                       const int32_t *d_intensity, const int32_t *d_bucketHead,
                                       const int32_t *d_nextPoint, int32_t queryIdx, int32_t queryX,
                                       int32_t queryY, int32_t queryZ, int32_t k, int32_t voxelSize,
                                       int32_t *distancesLocal, int32_t *intensitiesLocal) {
    for (int32_t j = 0; j < k; ++j) {
        distancesLocal[j] = INT32_MAX;
        intensitiesLocal[j] = -1;
    }
    int32_t actualK = 0;
    int32_t stopRadius = MAX_RADIUS;
    int32_t gx = queryX / voxelSize;
    int32_t gy = queryY / voxelSize;
    int32_t gz = queryZ / voxelSize;
    for (int32_t radius = 1; radius <= stopRadius; ++radius) {
        for (int32_t dx = -radius; dx <= radius; ++dx) {
            for (int32_t dy = -radius; dy <= radius; ++dy) {
                for (int32_t dz = -radius; dz <= radius; ++dz) {
                    if (dx > -radius && dx < radius && dy > -radius && dy < radius &&
                        dz > -radius && dz < radius)
                        continue;
                    uint32_t hash = getVoxelHashGPU(gx + dx, gy + dy, gz + dz);
                    int32_t bucketIdx = hash % HASH_TABLE_SIZE;
                    int32_t currPoint = d_bucketHead[bucketIdx];
                    while (currPoint != -1) {
                        if (currPoint != queryIdx) {
                            int32_t sqDist =
                                euclideanDistanceGPUInt(queryX, queryY, queryZ, d_x[currPoint],
                                                        d_y[currPoint], d_z[currPoint]);
                            if (sqDist < distancesLocal[k - 1]) {
                                int32_t pos = k - 1;
                                while (pos > 0 && distancesLocal[pos - 1] > sqDist) {
                                    distancesLocal[pos] = distancesLocal[pos - 1];
                                    intensitiesLocal[pos] = intensitiesLocal[pos - 1];
                                    pos--;
                                }
                                distancesLocal[pos] = sqDist;
                                intensitiesLocal[pos] = d_intensity[currPoint];
                                if (actualK < k)
                                    actualK++;
                            }
                        }
                        currPoint = d_nextPoint[currPoint];
                    }
                }
            }
        }
        if (actualK >= k && stopRadius == MAX_RADIUS)
            stopRadius = min(radius * 2, (int32_t)MAX_RADIUS);
    }
    return actualK;
}

__device__ int32_t findApproxKNNGPUFloat(const float *d_x, const float *d_y, const float *d_z,
                                         const int32_t *d_intensity, const int32_t *d_bucketHead,
                                         const int32_t *d_nextPoint, int32_t queryIdx, float queryX,
                                         float queryY, float queryZ, int32_t k, int32_t voxelSize,
                                         float *distancesLocal, int32_t *intensitiesLocal) {
    for (int32_t j = 0; j < k; ++j) {
        distancesLocal[j] = FLT_MAX;
        intensitiesLocal[j] = -1;
    }
    int32_t actualK = 0;
    int32_t stopRadius = MAX_RADIUS;
    int32_t gx = static_cast<int32_t>(queryX / voxelSize);
    int32_t gy = static_cast<int32_t>(queryY / voxelSize);
    int32_t gz = static_cast<int32_t>(queryZ / voxelSize);
    for (int32_t radius = 1; radius <= stopRadius; ++radius) {
        for (int32_t dx = -radius; dx <= radius; ++dx) {
            for (int32_t dy = -radius; dy <= radius; ++dy) {
                for (int32_t dz = -radius; dz <= radius; ++dz) {
                    if (dx > -radius && dx < radius && dy > -radius && dy < radius &&
                        dz > -radius && dz < radius)
                        continue;
                    uint32_t hash = getVoxelHashGPU(gx + dx, gy + dy, gz + dz);
                    int32_t bucketIdx = hash % HASH_TABLE_SIZE;
                    int32_t currPoint = d_bucketHead[bucketIdx];
                    while (currPoint != -1) {
                        if (currPoint != queryIdx) {
                            float sqDist =
                                euclideanDistanceGPUFloat(queryX, queryY, queryZ, d_x[currPoint],
                                                          d_y[currPoint], d_z[currPoint]);
                            if (sqDist < distancesLocal[k - 1]) {
                                int32_t pos = k - 1;
                                while (pos > 0 && distancesLocal[pos - 1] > sqDist) {
                                    distancesLocal[pos] = distancesLocal[pos - 1];
                                    intensitiesLocal[pos] = intensitiesLocal[pos - 1];
                                    pos--;
                                }
                                distancesLocal[pos] = sqDist;
                                intensitiesLocal[pos] = d_intensity[currPoint];
                                if (actualK < k)
                                    actualK++;
                            }
                        }
                        currPoint = d_nextPoint[currPoint];
                    }
                }
            }
        }
        if (actualK >= k && stopRadius == MAX_RADIUS)
            stopRadius = min(radius * 2, (int32_t)MAX_RADIUS);
    }
    return actualK;
}

__device__ int32_t computeCDFGPUInt(const int32_t *intensitiesLocal, int32_t actualK,
                                    int32_t centerIntensity, int32_t *cdfLocal) {
    int32_t histLocal[INTENSITY_LEVELS] = {0};
    histLocal[centerIntensity]++;
    for (int32_t j = 0; j < actualK; ++j) {
        if (intensitiesLocal[j] != -1)
            histLocal[intensitiesLocal[j]]++;
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

__device__ int32_t computeCDFGPUFloat(const int32_t *intensitiesLocal, int32_t actualK,
                                      int32_t centerIntensity, int32_t *cdfLocal) {
    int32_t histLocal[INTENSITY_LEVELS] = {0};
    histLocal[centerIntensity]++;
    for (int32_t j = 0; j < actualK; ++j) {
        if (intensitiesLocal[j] != -1)
            histLocal[intensitiesLocal[j]]++;
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

__global__ void buildGridKernelInt(const int32_t *d_x, const int32_t *d_y, const int32_t *d_z,
                                   int32_t *d_bucketHead, int32_t *d_nextPoint, int32_t numPoints,
                                   int32_t voxelSize) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    int32_t gx = d_x[i] / voxelSize;
    int32_t gy = d_y[i] / voxelSize;
    int32_t gz = d_z[i] / voxelSize;
    uint32_t hash = getVoxelHashGPU(gx, gy, gz);
    int32_t bucketIdx = hash % HASH_TABLE_SIZE;
    int32_t oldHead = atomicExch(&d_bucketHead[bucketIdx], i);
    d_nextPoint[i] = oldHead;
}

__global__ void buildGridKernelFloat(const float *d_x, const float *d_y, const float *d_z,
                                     int32_t *d_bucketHead, int32_t *d_nextPoint, int32_t numPoints,
                                     int32_t voxelSize) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    int32_t gx = static_cast<int32_t>(d_x[i] / voxelSize);
    int32_t gy = static_cast<int32_t>(d_y[i] / voxelSize);
    int32_t gz = static_cast<int32_t>(d_z[i] / voxelSize);
    uint32_t hash = getVoxelHashGPU(gx, gy, gz);
    int32_t bucketIdx = hash % HASH_TABLE_SIZE;
    int32_t oldHead = atomicExch(&d_bucketHead[bucketIdx], i);
    d_nextPoint[i] = oldHead;
}

__global__ void approxKnnKernelInt(const int32_t *d_x, const int32_t *d_y, const int32_t *d_z,
                                   const int32_t *d_intensity, int32_t *d_newIntensity,
                                   const int32_t *d_bucketHead, const int32_t *d_nextPoint,
                                   int32_t numPoints, int32_t k, int32_t voxelSize) {

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;

    int32_t distancesLocal[MAX_K];
    int32_t intensitiesLocal[MAX_K];

    int32_t actualK =
        findApproxKNNGPUInt(d_x, d_y, d_z, d_intensity, d_bucketHead, d_nextPoint, i, d_x[i],
                            d_y[i], d_z[i], k, voxelSize, distancesLocal, intensitiesLocal);

    if (actualK == 0) {
        d_newIntensity[i] = d_intensity[i];
        return;
    }

    int32_t originalIntensity = d_intensity[i];
    int32_t cdfLocal[INTENSITY_LEVELS];
    int32_t cdfMin = computeCDFGPUInt(intensitiesLocal, actualK, originalIntensity, cdfLocal);
    int32_t cdf = cdfLocal[originalIntensity];
    d_newIntensity[i] = computeIntensityGPU(originalIntensity, cdf, cdfMin, k + 1);
}

__global__ void approxKnnKernelFloat(const float *d_x, const float *d_y, const float *d_z,
                                     const int32_t *d_intensity, int32_t *d_newIntensity,
                                     const int32_t *d_bucketHead, const int32_t *d_nextPoint,
                                     int32_t numPoints, int32_t k, int32_t voxelSize) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints)
        return;
    float distancesLocal[MAX_K];
    int32_t intensitiesLocal[MAX_K];
    int32_t actualK =
        findApproxKNNGPUFloat(d_x, d_y, d_z, d_intensity, d_bucketHead, d_nextPoint, i, d_x[i],
                              d_y[i], d_z[i], k, voxelSize, distancesLocal, intensitiesLocal);
    if (actualK == 0) {
        d_newIntensity[i] = d_intensity[i];
        return;
    }
    int32_t originalIntensity = d_intensity[i];
    int32_t cdfLocal[INTENSITY_LEVELS];
    int32_t cdfMin = computeCDFGPUFloat(intensitiesLocal, actualK, originalIntensity, cdfLocal);
    int32_t cdf = cdfLocal[originalIntensity];
    d_newIntensity[i] = computeIntensityGPU(originalIntensity, cdf, cdfMin, k + 1);
}

float knnApproxGPUInt(const PointCloudInt &h_pc, int32_t k, int32_t *h_newIntensity) {
    int32_t numPoints = h_pc.numPoints;
    float voxelSize = calculateOptimalVoxelSize(h_pc);
    size_t intBytes = numPoints * sizeof(int32_t);
    int32_t *d_x, *d_y, *d_z, *d_intensity, *d_newIntensity;
    CUDA_CHECK(cudaMalloc(&d_x, intBytes));
    CUDA_CHECK(cudaMalloc(&d_y, intBytes));
    CUDA_CHECK(cudaMalloc(&d_z, intBytes));
    CUDA_CHECK(cudaMalloc(&d_intensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_newIntensity, intBytes));
    int32_t *d_bucketHead, *d_nextPoint;
    CUDA_CHECK(cudaMalloc(&d_bucketHead, HASH_TABLE_SIZE * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_nextPoint, intBytes));
    CUDA_CHECK(cudaMemset(d_bucketHead, -1, HASH_TABLE_SIZE * sizeof(int32_t)));
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
    buildGridKernelInt<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_bucketHead, d_nextPoint, numPoints,
                                                voxelSize);
    CUDA_CHECK_KERNEL();
    approxKnnKernelInt<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_intensity, d_newIntensity,
                                                d_bucketHead, d_nextPoint, numPoints, k, voxelSize);
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
    CUDA_CHECK(cudaFree(d_bucketHead));
    CUDA_CHECK(cudaFree(d_nextPoint));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

float knnApproxGPUFloat(const PointCloudFloat &h_pc, int32_t k, int32_t *h_newIntensity) {
    int32_t numPoints = h_pc.numPoints;
    float voxelSize = calculateOptimalVoxelSize(h_pc);
    size_t fBytes = numPoints * sizeof(float);
    size_t intBytes = numPoints * sizeof(int32_t);
    float *d_x, *d_y, *d_z;
    int32_t *d_intensity, *d_newIntensity;
    CUDA_CHECK(cudaMalloc(&d_x, fBytes));
    CUDA_CHECK(cudaMalloc(&d_y, fBytes));
    CUDA_CHECK(cudaMalloc(&d_z, fBytes));
    CUDA_CHECK(cudaMalloc(&d_intensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_newIntensity, intBytes));
    int32_t *d_bucketHead, *d_nextPoint;
    CUDA_CHECK(cudaMalloc(&d_bucketHead, HASH_TABLE_SIZE * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_nextPoint, intBytes));
    CUDA_CHECK(cudaMemset(d_bucketHead, -1, HASH_TABLE_SIZE * sizeof(int32_t)));
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
    buildGridKernelFloat<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_bucketHead, d_nextPoint,
                                                  numPoints, voxelSize);
    CUDA_CHECK_KERNEL();
    approxKnnKernelFloat<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_intensity, d_newIntensity,
                                                  d_bucketHead, d_nextPoint, numPoints, k,
                                                  voxelSize);
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
    CUDA_CHECK(cudaFree(d_bucketHead));
    CUDA_CHECK(cudaFree(d_nextPoint));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

} // namespace knnApprox