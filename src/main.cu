#include "common.cuh"
#include "kMeans.cuh"
#include "knn.cuh"
#include "knnApprox.cuh"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./a2 <input_file.txt> <knn|approx_knn|kmeans>\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string algo = argv[2];

    PointCloudInt pc;
    int32_t k, T;

    if (!readInputFileInt(inputFile.c_str(), pc, k, T)) {
        return 1;
    }

    int32_t *out = new int32_t[pc.numPoints];
    std::string outputFile;

    if (algo == "knn") {
        knn::knnGPUInt(pc, k, out);
        outputFile = "knn.txt";
    } else if (algo == "approx_knn") {
        knnApprox::knnApproxGPUInt(pc, k, out);
        outputFile = "approx_knn.txt";
    } else if (algo == "kmeans") {
        kMeans::kMeansGPUInt(pc, k, T, out);
        outputFile = "kmeans.txt";
    } else {
        std::cerr << "Unknown algorithm: " << algo << "\n";
        delete[] out;
        freePointCloudInt(pc);
        return 1;
    }

    if (!writeOutputFileInt(outputFile.c_str(), pc, out)) {
        std::cerr << "Failed to write output to " << outputFile << "\n";
    }

    delete[] out;
    freePointCloudInt(pc);

    return 0;
}
