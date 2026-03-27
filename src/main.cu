#include "common.cuh"
#include "kMeans.cuh"
#include "knn.cuh"
#include "knnApprox.cuh"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./a2 <input_file.txt>\n";
        return 1;
    }

    PointCloudInt pc;
    int32_t k, T;

    if (!readInputFileInt(argv[1], pc, k, T)) {
        return 1;
    }

    int32_t *outKnn = new int32_t[pc.numPoints];
    int32_t *outKnnApprox = new int32_t[pc.numPoints];
    int32_t *outKMeans = new int32_t[pc.numPoints];

    knn::knnGPUInt(pc, k, outKnn);
    knnApprox::knnApproxGPUInt(pc, k, outKnnApprox);
    kMeans::kMeansGPUInt(pc, k, T, outKMeans);

    writeOutputFileInt("knn.txt", pc, outKnn);
    writeOutputFileInt("approx.knn.txt", pc, outKnnApprox);
    writeOutputFileInt("kmeans.txt", pc, outKMeans);

    delete[] outKnn;
    delete[] outKnnApprox;
    delete[] outKMeans;
    freePointCloudInt(pc);

    return 0;
}
