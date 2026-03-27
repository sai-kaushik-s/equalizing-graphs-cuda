#include "common.cuh"
#include "kMeans.cuh"
#include "knn.cuh"
#include "knnApprox.cuh"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sys/stat.h>

void createDirectories() {
    mkdir("output", 0755);
    mkdir("output/knn", 0755);
    mkdir("output/approx_knn", 0755);
    mkdir("output/kmeans", 0755);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <int_input.txt> <float_input.txt> <knn|approx_knn|kmeans>\n";
        return 1;
    }

    std::string intInput = argv[1];
    std::string floatInput = argv[2];
    std::string algo = argv[3];

    createDirectories();

    PointCloudInt pcInt;
    PointCloudFloat pcFloat;
    int32_t k, T;

    if (!readInputFileInt(intInput.c_str(), pcInt, k, T)) {
        return 1;
    }
    if (!readInputFileFloat(floatInput.c_str(), pcFloat, k, T)) {
        freePointCloudInt(pcInt);
        return 1;
    }

    int32_t *outInt = new int32_t[pcInt.numPoints];
    int32_t *outFloat = new int32_t[pcFloat.numPoints];
    int32_t *outCpuInt = new int32_t[pcInt.numPoints];
    int32_t *outCpuFloat = new int32_t[pcFloat.numPoints];
    
    float durationGpuInt = 0.0f, durationCpuInt = 0.0f;
    float durationGpuFloat = 0.0f, durationCpuFloat = 0.0f;
    std::string outPathInt, outPathFloat;

    if (algo == "knn") {
        durationGpuInt = knn::knnGPUInt(pcInt, k, outInt);
        // durationCpuInt = knn::knnCPUInt(pcInt, k, outCpuInt);
        durationGpuFloat = knn::knnGPUFloat(pcFloat, k, outFloat);
        // durationCpuFloat = knn::knnCPUFloat(pcFloat, k, outCpuFloat);
        
        outPathInt = "output/knn/n_" + std::to_string(pcInt.numPoints) + "_k_" + std::to_string(k) + "_int.txt";
        outPathFloat = "output/knn/n_" + std::to_string(pcFloat.numPoints) + "_k_" + std::to_string(k) + "_float.txt";
    } else if (algo == "approx_knn") {
        durationGpuInt = knnApprox::knnApproxGPUInt(pcInt, k, outInt);
        // durationCpuInt = knnApprox::knnApproxCPUInt(pcInt, k, outCpuInt);
        durationGpuFloat = knnApprox::knnApproxGPUFloat(pcFloat, k, outFloat);
        // durationCpuFloat = knnApprox::knnApproxCPUFloat(pcFloat, k, outCpuFloat);
        
        outPathInt = "output/approx_knn/n_" + std::to_string(pcInt.numPoints) + "_k_" + std::to_string(k) + "_int.txt";
        outPathFloat = "output/approx_knn/n_" + std::to_string(pcFloat.numPoints) + "_k_" + std::to_string(k) + "_float.txt";
    } else if (algo == "kmeans") {
        durationGpuInt = kMeans::kMeansGPUInt(pcInt, k, T, outInt);
        // durationCpuInt = kMeans::kMeansCPUInt(pcInt, k, T, outCpuInt);
        durationGpuFloat = kMeans::kMeansGPUFloat(pcFloat, k, T, outFloat);
        // durationCpuFloat = kMeans::kMeansCPUFloat(pcFloat, k, T, outCpuFloat);
        
        outPathInt = "output/kmeans/n_" + std::to_string(pcInt.numPoints) + "_k_" + std::to_string(k) + "_t_" + std::to_string(T) + "_int.txt";
        outPathFloat = "output/kmeans/n_" + std::to_string(pcFloat.numPoints) + "_k_" + std::to_string(k) + "_t_" + std::to_string(T) + "_float.txt";
    } else {
        std::cerr << "Unknown algorithm: " << algo << "\n";
        delete[] outInt; delete[] outFloat; delete[] outCpuInt; delete[] outCpuFloat;
        freePointCloudInt(pcInt); freePointCloudFloat(pcFloat);
        return 1;
    }

    std::cout << "Algorithm: " << algo << "\n";
    std::cout << "GPU Time (INT32): " << formatDuration(durationGpuInt) << "\n";
    // std::cout << "CPU Time (INT32): " << formatDuration(durationCpuInt) << "\n";
    std::cout << "GPU Time (FLOAT): " << formatDuration(durationGpuFloat) << "\n";
    // std::cout << "CPU Time (FLOAT): " << formatDuration(durationCpuFloat) << "\n";

    if (!writeOutputFileInt(outPathInt.c_str(), pcInt, outInt)) {
        std::cerr << "Failed to write INT output: " << outPathInt << "\n";
    } else {
        std::cout << "Saved INT output to: " << outPathInt << "\n";
    }

    if (!writeOutputFileFloat(outPathFloat.c_str(), pcFloat, outFloat)) {
        std::cerr << "Failed to write FLOAT output: " << outPathFloat << "\n";
    } else {
        std::cout << "Saved FLOAT output to: " << outPathFloat << "\n";
    }

    delete[] outInt; delete[] outFloat; delete[] outCpuInt; delete[] outCpuFloat;
    freePointCloudInt(pcInt); freePointCloudFloat(pcFloat);

    return 0;
}
