#include <filesystem>
#include <fstream>
#include <iostream>

#include "common.cuh"
#include "knn.cuh"
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./a2 <input_file.txt>\n";
        return 1;
    }

    PointCloud pc;
    int k, T;

    if (!readInputFile(argv[1], pc, k, T)) {
        return 1;
    }

    std::string dir = "output/n_" + std::to_string(pc.numPoints) +
                  "_k_" + std::to_string(k) +
                  "_T_" + std::to_string(T);
    std::filesystem::create_directories(dir);

    Logger logger(dir + "/run.log");

    std::cout << "Reading input file...\n";
    std::cout << "Loaded " << pc.numPoints << " points. (k=" << k << ", T=" << T << ")\n\n";

    int* outKnnGPU = new int[pc.numPoints];

    auto knnGPUDuration = timeFunction("GPU KNN", knnGPU, pc, k, outKnnGPU);
    writeOutputFile(dir + "/knn.txt", pc, outKnnGPU);

    delete[] outKnnGPU;
    freePointCloud(pc);

    return 0;
}