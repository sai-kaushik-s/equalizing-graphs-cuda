#pragma once
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#define INTENSITY_LEVELS 256
#define MAX_K 128

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA API Error at " << __FILE__ << ":" << __LINE__ << " - "              \
                      << cudaGetErrorString(err) << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                                        \
    do {                                                                                           \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA Kernel Launch Error at " << __FILE__ << ":" << __LINE__ << " - "    \
                      << cudaGetErrorString(err) << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
        err = cudaDeviceSynchronize();                                                             \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA Kernel Sync Error at " << __FILE__ << ":" << __LINE__ << " - "      \
                      << cudaGetErrorString(err) << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#include <cstdint>

struct PointCloudFloat {
    float *x;
    float *y;
    float *z;
    int32_t *intensity;
    int32_t numPoints;
};

struct PointCloudInt {
    int32_t *x;
    int32_t *y;
    int32_t *z;
    int32_t *intensity;
    int32_t numPoints;
};

inline int32_t euclideanDistanceCPUInt(int32_t x1, int32_t y1, int32_t z1, int32_t x2, int32_t y2,
                                       int32_t z2) {
    int32_t dx = x1 - x2;
    int32_t dy = y1 - y2;
    int32_t dz = z1 - z2;
    return (dx * dx) + (dy * dy) + (dz * dz);
}

inline float euclideanDistanceCPUFloat(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return (dx * dx) + (dy * dy) + (dz * dz);
}

__device__ __forceinline__ int32_t euclideanDistanceGPUInt(int32_t x1, int32_t y1, int32_t z1,
                                                           int32_t x2, int32_t y2, int32_t z2) {
    int32_t dx = x1 - x2;
    int32_t dy = y1 - y2;
    int32_t dz = z1 - z2;
    return (dx * dx) + (dy * dy) + (dz * dz);
}

__device__ __forceinline__ float euclideanDistanceGPUFloat(float x1, float y1, float z1, float x2,
                                                           float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return (dx * dx) + (dy * dy) + (dz * dz);
}

inline int32_t computeIntensityCPU(int32_t intensity, int32_t cdf, int32_t cdfMin, int32_t m) {
    if (m == cdfMin)
        return intensity;
    return static_cast<int32_t>((static_cast<int64_t>(cdf - cdfMin) * (INTENSITY_LEVELS - 1)) /
                                (m - cdfMin));
}

__device__ __forceinline__ int32_t computeIntensityGPU(int32_t intensity, int32_t cdf,
                                                       int32_t cdfMin, int32_t m) {
    if (m == cdfMin)
        return intensity;
    return static_cast<int32_t>((static_cast<int64_t>(cdf - cdfMin) * (INTENSITY_LEVELS - 1)) /
                                (m - cdfMin));
}

void allocatePointCloudInt(PointCloudInt &pc, int32_t n);
void freePointCloudInt(PointCloudInt &pc);
bool readInputFileInt(const std::string &filename, PointCloudInt &pc, int32_t &k, int32_t &T);
bool writeOutputFileInt(const std::string &filename, const PointCloudInt &pc,
                        const int32_t *newIntensities);
float calculateMAEInt(const int32_t *exact, const int32_t *approx, int32_t numPoints);

void allocatePointCloudFloat(PointCloudFloat &pc, int32_t n);
void freePointCloudFloat(PointCloudFloat &pc);
bool readInputFileFloat(const std::string &filename, PointCloudFloat &pc, int32_t &k, int32_t &T);
bool writeOutputFileFloat(const std::string &filename, const PointCloudFloat &pc,
                          const int32_t *newIntensities);
float calculateMAEFloat(const int32_t *exact, const int32_t *approx, int32_t numPoints);

std::string formatDuration(float ms);