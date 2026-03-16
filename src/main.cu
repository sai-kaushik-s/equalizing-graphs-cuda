#include <iostream>
#include <fstream>
#include <vector>
#include "common.cuh"
#include "knn.cuh"
#include "approx_knn.cuh"
#include "kmeans.cuh"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./a2 <input_file.txt>\n";
        return 1;
    }

    return 0;
}