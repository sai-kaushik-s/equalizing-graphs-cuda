#include "common.cuh"
#include "kMeans.cuh"
#include "knn.cuh"
#include "knnApprox.cuh"

#include <chrono>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: bin/compare <int_input_file.txt> <float_input_file.txt>\n";
        return 1;
    }

    PointCloudInt pcInt;
    PointCloudFloat pcFloat;
    int32_t k, T;

    if (!readInputFileInt(argv[1], pcInt, k, T))
        return 1;
    if (!readInputFileFloat(argv[2], pcFloat, k, T))
        return 1;

    std::cout << "Loaded " << pcInt.numPoints << " points. (k=" << k << ", T=" << T << ")\n\n";

    int32_t *outKnnGPUFloat = new int32_t[pcFloat.numPoints];
    int32_t *outKnnApproxGPUFloat = new int32_t[pcFloat.numPoints];
    int32_t *outKMeansGPUFloat = new int32_t[pcFloat.numPoints];

    int32_t *outKnnCPUFloat = new int32_t[pcFloat.numPoints];
    int32_t *outKnnApproxCPUFloat = new int32_t[pcFloat.numPoints];
    int32_t *outKMeansCPUFloat = new int32_t[pcFloat.numPoints];

    int32_t *outKnnGPUInt = new int32_t[pcInt.numPoints];
    int32_t *outKnnApproxGPUInt = new int32_t[pcInt.numPoints];
    int32_t *outKMeansGPUInt = new int32_t[pcInt.numPoints];

    int32_t *outKnnCPUInt = new int32_t[pcInt.numPoints];
    int32_t *outKnnApproxCPUInt = new int32_t[pcInt.numPoints];
    int32_t *outKMeansCPUInt = new int32_t[pcInt.numPoints];

    std::cout << "--- FLOAT Versions ---\n";
    float tKnnGF = knn::knnGPUFloat(pcFloat, k, outKnnGPUFloat);
    float tKnnCF = knn::knnCPUFloat(pcFloat, k, outKnnCPUFloat);
    std::cout << "KNN: GPU=" << formatDuration(tKnnGF) << ", CPU=" << formatDuration(tKnnCF)
              << " (Speedup: " << (tKnnCF / tKnnGF) << "x)\n";

    float tKnnAGF = knnApprox::knnApproxGPUFloat(pcFloat, k, outKnnApproxGPUFloat);
    float tKnnACF = knnApprox::knnApproxCPUFloat(pcFloat, k, outKnnApproxCPUFloat);
    std::cout << "Approx KNN: GPU=" << formatDuration(tKnnAGF)
              << ", CPU=" << formatDuration(tKnnACF) << " (Speedup: " << (tKnnACF / tKnnAGF)
              << "x)\n";

    float tKMGF = kMeans::kMeansGPUFloat(pcFloat, k, T, outKMeansGPUFloat);
    float tKMCF = kMeans::kMeansCPUFloat(pcFloat, k, T, outKMeansCPUFloat);
    std::cout << "K-Means: GPU=" << formatDuration(tKMGF) << ", CPU=" << formatDuration(tKMCF)
              << " (Speedup: " << (tKMCF / tKMGF) << "x)\n";

    std::cout << "\n--- FLOAT Metrics ---\n";
    std::cout << "[GPU] Approx KNN MAE: "
              << calculateMAEFloat(outKnnGPUFloat, outKnnApproxGPUFloat, pcFloat.numPoints) << "\n";
    std::cout << "[CPU] Approx KNN MAE: "
              << calculateMAEFloat(outKnnCPUFloat, outKnnApproxCPUFloat, pcFloat.numPoints) << "\n";
    std::cout << "[GPU] K-Means MAE:    "
              << calculateMAEFloat(outKnnGPUFloat, outKMeansGPUFloat, pcFloat.numPoints) << "\n";
    std::cout << "[CPU] K-Means MAE:    "
              << calculateMAEFloat(outKnnCPUFloat, outKMeansCPUFloat, pcFloat.numPoints) << "\n";

    std::cout << "\n--- INT32 Versions ---\n";
    float tKnnGI = knn::knnGPUInt(pcInt, k, outKnnGPUInt);
    float tKnnCI = knn::knnCPUInt(pcInt, k, outKnnCPUInt);
    std::cout << "KNN: GPU=" << formatDuration(tKnnGI) << ", CPU=" << formatDuration(tKnnCI)
              << " (Speedup: " << (tKnnCI / tKnnGI) << "x)\n";

    float tKnnAGI = knnApprox::knnApproxGPUInt(pcInt, k, outKnnApproxGPUInt);
    float tKnnACI = knnApprox::knnApproxCPUInt(pcInt, k, outKnnApproxCPUInt);
    std::cout << "Approx KNN: GPU=" << formatDuration(tKnnAGI)
              << ", CPU=" << formatDuration(tKnnACI) << " (Speedup: " << (tKnnACI / tKnnAGI)
              << "x)\n";

    float tKMGI = kMeans::kMeansGPUInt(pcInt, k, T, outKMeansGPUInt);
    float tKMCI = kMeans::kMeansCPUInt(pcInt, k, T, outKMeansCPUInt);
    std::cout << "K-Means: GPU=" << formatDuration(tKMGI) << ", CPU=" << formatDuration(tKMCI)
              << " (Speedup: " << (tKMCI / tKMGI) << "x)\n";

    std::cout << "\n--- INT32 Metrics ---\n";
    std::cout << "[GPU] Approx KNN MAE: "
              << calculateMAEInt(outKnnGPUInt, outKnnApproxGPUInt, pcInt.numPoints) << "\n";
    std::cout << "[CPU] Approx KNN MAE: "
              << calculateMAEInt(outKnnCPUInt, outKnnApproxCPUInt, pcInt.numPoints) << "\n";
    std::cout << "[GPU] K-Means MAE:    "
              << calculateMAEInt(outKnnGPUInt, outKMeansGPUInt, pcInt.numPoints) << "\n";
    std::cout << "[CPU] K-Means MAE:    "
              << calculateMAEInt(outKnnCPUInt, outKMeansCPUInt, pcInt.numPoints) << "\n";

    delete[] outKnnGPUFloat;
    delete[] outKnnApproxGPUFloat;
    delete[] outKMeansGPUFloat;
    delete[] outKnnCPUFloat;
    delete[] outKnnApproxCPUFloat;
    delete[] outKMeansCPUFloat;

    delete[] outKnnGPUInt;
    delete[] outKnnApproxGPUInt;
    delete[] outKMeansGPUInt;
    delete[] outKnnCPUInt;
    delete[] outKnnApproxCPUInt;
    delete[] outKMeansCPUInt;

    freePointCloudInt(pcInt);
    freePointCloudFloat(pcFloat);

    return 0;
}