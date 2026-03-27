# In-Depth Project Documentation: Equalizing Graphs CUDA

This document provides a detailed, technical explanation for every function, member function, and CUDA kernel in the `equalizing-graphs-cuda` project. Each section covers implementation logic, performance considerations, and role within the overall intensity equalization pipeline.

---

## 1. Common Utilities ([include/common.cuh](include/common.cuh), [src/common.cu](src/common.cu))

### Math & Geometry Functions
- **`euclideanDistanceCPUInt(x1, y1, z1, x2, y2, z2)`**: Calculates the squared Euclidean distance between two 3D points. It avoids a costly square root operation, which is sufficient for comparisons and sorting (e.g., finding the "nearest" point). Uses `int32_t` for intermediate calculations.
- **`euclideanDistanceCPUFloat(x1, y1, z1, x2, y2, z2)`**: Floating-point version of the squared distance calculation. Essential for high-precision point clouds.
- **`euclideanDistanceGPUInt(...)`**: A `__device__ __forceinline__` function providing high-performance squared distance calculations within CUDA kernels. Inlining reduces function call overhead in hot loops.
- **`euclideanDistanceGPUFloat(...)`**: Floating-point GPU version of the distance function, optimized for register-heavy CUDA threads.

### Intensity Transformation
- **`computeIntensityCPU(intensity, cdf, cdfMin, m)`**: Implements the histogram equalization formula. It maps the current intensity $v$ to $h(v) = \text{round}( \frac{cdf(v) - cdf_{min}}{m - cdf_{min}} \times 255 )$. It handles the edge case where $m = cdf_{min}$ by returning the original intensity to avoid division by zero. Intermediate products use `int64_t` to prevent overflow during the multiplication by 255.
- **`computeIntensityGPU(...)`**: The device-side equivalent for intensity remapping. It mirrors the CPU logic exactly to ensure parity between host and device results.

### Memory & File I/O
- **`allocatePointCloudInt(pc, n)`**: Dynamically allocates four separate arrays ($x, y, z, intensity$) of size $n$ on the host. This Structure of Arrays (SoA) format is preferred for GPU computing as it enables coalesced memory access patterns.
- **`freePointCloudInt(pc)`**: Safely deletes the dynamically allocated arrays in a `PointCloudInt` structure to prevent memory leaks.
- **`readInputFileInt(filename, pc, k, T)`**: Parses input files containing a header ($n, k, T$) followed by $n$ lines of coordinates and intensities. It populates the provided `pc` structure and returns parameters by reference.
- **`writeOutputFileInt(filename, pc, newIntensities)`**: Writes the original $x, y, z$ coordinates paired with the newly calculated intensities to the output file.
- **`calculateMAEInt(exact, approx, n)`**: Computes the Mean Absolute Error between an exact dataset (from KNN) and an approximate one (from Approx-KNN or k-Means). Uses OpenMP `#pragma omp parallel for reduction(+:totalError)` for fast parallel processing.
- **`formatDuration(ms)`**: Converts raw millisecond timings into more readable strings. It automatically switches units (ns, µs, ms, s) based on the magnitude of the duration and uses broad precision for clear output.

---

## 2. Exact K-Nearest Neighbors ([include/knn.cuh](include/knn.cuh), [src/knn.cu](src/knn.cu))

### CPU implementation (OpenMP)
- **`knn::findKNNCPUInt(...)`**: Finds the $k$ nearest neighbors for a query point. It utilizes `std::nth_element`, which provides $O(n)$ average complexity per point. It sorts the distance buffer just enough to separate the $k$ closest elements from the rest.
- **`knn::computeCDFCPUInt(...)`**: Builds an intensity histogram from the $k$ neighbors plus the query point itself (total $k+1$ points). It then generates the CDF used for remapping that specific point.
- **`knn::knnCPUInt(...)`**: Orchestrates the per-point KNN loop. By using thread-local vectors for neighbor calculations, it minimizes allocation overhead and maximizes CPU core utilization through OpenMP.

### GPU kernels & Device Functions
- **`knn::findKNNGPUInt(...)`**: A device-side search function. Since each thread has limited memory, it avoids sorting the entire point cloud. Instead, it maintains a small fixed-size sorted array of the current $k$-best neighbors, resulting in an $O(n \cdot k)$ complexity.
- **`knn::computeCDFGPU(...)`**: Device-side histogram and CDF calculation. It integrates the intensities of the $k$ neighbors into a local array in registers or local memory to avoid global synchronization.
- **`knn::knnKernelInt(...)`**: The primary entry point for GPU KNN. Each CUDA thread calculates the full neighborhood for one point. It uses `MAX_K` as a compile-time limit (typically 128) to allow for efficient stack allocation of local distance/intensity arrays.

---

## 3. Approximate K-Nearest Neighbors ([include/knnApprox.cuh](include/knnApprox.cuh), [src/knnApprox.cu](src/knnApprox.cu))

### Voxel Grid Heuristics
- **`knnApprox::calculateOptimalVoxelSizeImpl(...)`**: Estimates the side length of a cube (voxel) that would result in a small, balanced number of points per cell on average. It calculates the volume of the point cloud's bounding box and computes the cubed root of the volume-to-density ratio.
- **`knnApprox::getVoxelHash(...)`**: A spatial hashing function that maps $(gx, gy, gz)$ grid coordinates to a 64-bit integer using high-quality prime multipliers to minimize collisions in the voxel lookup table.

### CPU implementation (OpenMP)
- **`knnApprox::knnApproxCPUInt(...)`**: Top-level function that builds an `std::unordered_map` voxel grid and parallelizes the expanding search across cores.

### GPU kernels & Device Functions
- **`knnApprox::buildGridKernelInt(...)`**: A specialized kernel for constructing a spatial hash table on the GPU. It uses `atomicExch` to create a per-bucket linked list of point indices. This structure allows the GPU to quickly find all points within a specific spatial voxel without needing a structured 3D array.
- **`knnApprox::findApproxKNNGPUInt(...)`**: The device-side version of the expanding search. It navigates the spatial hash table by visiting hashes of neighboring grid cells and traversing the linked lists found in those buckets.
- **`knnApprox::approxKnnKernelInt(...)`**: A per-point kernel that combines spatial grid navigation, candidate collection, and intensity remapping in a single efficient pass.

---

## 4. k-Means Clustering ([include/kMeans.cuh](include/kMeans.cuh), [src/kMeans.cu](src/kMeans.cu))

### CPU implementation (OpenMP)
- **`kMeans::assignClustersCPUInt(...)`**: Parallelized loop that assigns each point to its nearest centroid. It includes deterministic tie-breaking logic (checking coordinates lexicographically) to ensure consistent results across platforms if multiple centroids are equidistant.
- **`kMeans::updateCentroidsCPUInt(...)`**: Calculates new cluster centers. It uses local thread-private accumulators summed into global counters using a `critical` section to avoid race conditions while maintaining high performance.
- **`kMeans::computeClusterCDFsCPUInt(...)`**: First, it builds a global 2D histogram where each row corresponds to a cluster. Then, it sequentially computes the Cumulative Distribution Function (CDF) for each cluster by integrating its histogram row and identifying the first non-zero entry ($cdf_{min}$).
- **`kMeans::kMeansCPUInt(...)`**: Coordinates the entire k-Means pipeline on the CPU. It runs the assignment/update loop for $T$ iterations and then applies the final intensity remapping based on the cluster-specific CDFs.

### GPU Kernels (`__global__`) & Device Functions
- **`kMeans::assignClustersGPUInt(...)`**: High-performance cluster assignment kernel. It uses **shared memory** to load all centroids once per block, drastically reducing the number of global memory reads needed as threads compare points to every centroid.
- **`kMeans::accumulateCentroidsGPUInt(...)`**: Uses `atomicAdd` on global memory to sum the $x, y, z$ coordinates and counts for each cluster. It uses local partial sums in shared memory to minimize contention on the atomic counters.
- **`kMeans::updateCentroidsGPUInt(...)`**: A simple 1-to-1 kernel where each thread calculates the new center for exactly one cluster by dividing the accumulated sums by the point counts.
- **`kMeans::computeHistogramsGPUInt(...)`**: Utilizes global `atomicAdd` to increment intensity bins for each cluster. This kernel transforms the point cloud into a compact histogram representation.
- **`kMeans::computeCDFsGPUInt(...)`**: Parallel kernel where each thread processes a single cluster. It iterates through the 256 intensity bins to compute a local prefix sum (the CDF) and determines $cdf_{min}$ for that cluster.
- **`kMeans::remapIntensityGPUInt(...)`**: Final transformation kernel. Each thread fetches its point's cluster index and original intensity, then looks up the corresponding CDF value in global memory to perform the equalization formula.

### GPU Host Wrappers
- **`kMeans::kMeansGPUInt(...)`**: Manages the host-to-device data flow. It handles `cudaMalloc`, `cudaMemset`, and `cudaMemcpy` operations. It also implements an **early convergence** check by copying a "changed" flag from the device; if no points change clusters in an iteration, the algorithm stops early to save time.

---

## 5. Main Runners ([src/main.cu](src/main.cu), [src/compare.cu](src/compare.cu))

- **`main(...)` in [main.cu](src/main.cu)**: The primary executable driver for the assignment. It loads a dataset, runs the GPU versions of all three algorithms sequentially, and outputs the individual results to separate TXT files.
- **`main(...)` in [compare.cu](src/compare.cu)**: A benchmarking and verification tool. It runs both CPU and GPU implementations, measures speedups, and calculates accuracy metrics (MAE) for approximate methods (Approx-KNN and k-Means) by comparing them to the "ground truth" results from exact KNN.
