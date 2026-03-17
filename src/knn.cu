#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>

#include "knn.cuh"

#define MAX_K 256
#define INTENSITY_LEVELS 256

struct NeighborDistance {
    float distance;
    int pointIdx;

    bool operator<(const NeighborDistance& other) const {
        return distance < other.distance;
    }
};

inline float euclideanDistance(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return (dx * dx) + (dy * dy) + (dz * dz);
}

void findKNN(const PointCloud& pc, int queryIdx, int k, 
             std::vector<NeighborDistance>& neighborDistances) {
    
    float queryX = pc.x[queryIdx];
    float queryY = pc.y[queryIdx];
    float queryZ = pc.z[queryIdx];

    for (int i = 0; i < pc.numPoints; ++i) {
        float sqDist = euclideanDistance(queryX, queryY, queryZ, pc.x[i], pc.y[i], pc.z[i]);
        neighborDistances[i] = {sqDist, i};
    }

    std::nth_element(neighborDistances.begin(), neighborDistances.begin() + k, neighborDistances.end());
}

int computeCDF(const PointCloud& pc, const std::vector<NeighborDistance>& neighbors, int k, int* cdfLocal) {
    int histLocal[INTENSITY_LEVELS] = {0};

    for (int i = 0; i < k; ++i) {
        int neighborIdx = neighbors[i].pointIdx;
        int intensity = pc.intensity[neighborIdx];
        histLocal[intensity]++;
    }

    cdfLocal[0] = histLocal[0];
    for (int v = 1; v < INTENSITY_LEVELS; ++v) {
        cdfLocal[v] = cdfLocal[v - 1] + histLocal[v];
    }

    int cdfMin = -1;
    for (int v = 0; v < INTENSITY_LEVELS; ++v) {
        if (cdfLocal[v] > 0) {
            cdfMin = cdfLocal[v];
            break;
        }
    }
    return cdfMin;
}

int computeIntensity(int intensity, int cdf, int cdfMin, int m) {
    if (m == cdfMin) {
        return intensity;
    }

    float num = static_cast<float>(cdf - cdfMin);
    float den = static_cast<float>(m - cdfMin);
    
    return static_cast<int>(std::floor((num / den) * (INTENSITY_LEVELS - 1)));
}

void knnCPU(const PointCloud& pc, int k, int* newIntensity) {
    #pragma omp parallel
    {
        std::vector<NeighborDistance> distancesLocal(pc.numPoints);
        int cdfLocal[INTENSITY_LEVELS];

        #pragma omp for
        for (int i = 0; i < pc.numPoints; ++i) {
            findKNN(pc, i, k, distancesLocal);

            int cdfMin = computeCDF(pc, distancesLocal, k, cdfLocal);

            int intensity = pc.intensity[i];
            int cdf = cdfLocal[intensity];
            
            newIntensity[i] = computeIntensity(intensity, cdf, cdfMin, k);
        }
    }
}

__device__ __forceinline__ float euclideanDistanceGPU(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return (dx * dx) + (dy * dy) + (dz * dz);
}

__device__ void findKNNGPU(const float* d_x, const float* d_y, const float* d_z, const int* d_intensity,
                           int numPoints, float queryX, float queryY, float queryZ, int k,
                           float* distancesLocal, int* intensitiesLocal) {
    for (int i = 0; i < k; ++i) {
        distancesLocal[i] = 1e30f; 
        intensitiesLocal[i] = -1;
    }

    for (int i = 0; i < numPoints; ++i) {
        float sqDist = euclideanDistanceGPU(queryX, queryY, queryZ, d_x[i], d_y[i], d_z[i]);

        if (sqDist < distancesLocal[k - 1]) {
            int pos = k - 1;
            while (pos > 0 && distancesLocal[pos - 1] > sqDist) {
                distancesLocal[pos] = distancesLocal[pos - 1];
                intensitiesLocal[pos] = intensitiesLocal[pos - 1];
                pos--;
            }
            distancesLocal[pos] = sqDist;
            intensitiesLocal[pos] = d_intensity[i];
        }
    }
}

__device__ int computeCDFGPU(const int* intensitiesLocal, int k, int* cdfLocal) {
    int histLocal[INTENSITY_LEVELS] = {0};
    for (int i = 0; i < k; ++i) {
        if (intensitiesLocal[i] != -1) {
            histLocal[intensitiesLocal[i]]++;
        }
    }

    cdfLocal[0] = histLocal[0];
    for (int v = 1; v < INTENSITY_LEVELS; ++v) {
        cdfLocal[v] = cdfLocal[v - 1] + histLocal[v];
    }

    int cdfMin = -1;
    for (int v = 0; v < INTENSITY_LEVELS; ++v) {
        if (cdfLocal[v] > 0) {
            cdfMin = cdfLocal[v];
            break;
        }
    }
    return cdfMin;
}

__device__ __forceinline__ int computeIntensityGPU(int intensity, int cdf, int cdfMin, int m) {
    if (m == cdfMin) {
        return intensity;
    }
    
    float num = static_cast<float>(cdf - cdfMin);
    float den = static_cast<float>(m - cdfMin);
    
    return static_cast<int>(floorf((num / den) * (INTENSITY_LEVELS - 1)));
}

__global__ void knnKernel(const float* d_x, const float* d_y, const float* d_z, 
                          const int* d_intensity, int* d_newIntensity, 
                          int numPoints, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;

    float distancesLocal[MAX_K];
    int intensitiesLocal[MAX_K];

    findKNNGPU(
        d_x, d_y, d_z, d_intensity, numPoints,
        d_x[i], d_y[i], d_z[i], k,
        distancesLocal, intensitiesLocal
    );

    int cdfLocal[INTENSITY_LEVELS];
    int cdfMin = computeCDFGPU(intensitiesLocal, k, cdfLocal);

    int intensity = d_intensity[i];
    int cdf = cdfLocal[intensity];
    
    d_newIntensity[i] = computeIntensityGPU(
        intensity, 
        cdf, 
        cdfMin, 
        k
    );
}

void knnGPU(const PointCloud& h_pc, int k, int* h_newIntensity) {
    int numPoints = h_pc.numPoints;
    size_t floatBytes = numPoints * sizeof(float);
    size_t intBytes = numPoints * sizeof(int);

    float *d_x, *d_y, *d_z;
    int *d_intensity, *d_newIntensity;
    
    CUDA_CHECK(cudaMalloc(&d_x, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_y, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_z, floatBytes));
    CUDA_CHECK(cudaMalloc(&d_intensity, intBytes));
    CUDA_CHECK(cudaMalloc(&d_newIntensity, intBytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_pc.x, floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_pc.y, floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_pc.z, floatBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intensity, h_pc.intensity, intBytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    
    knnKernel<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_intensity, d_newIntensity, numPoints, k);
    
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_newIntensity, d_newIntensity, intBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_intensity));
    CUDA_CHECK(cudaFree(d_newIntensity));
}