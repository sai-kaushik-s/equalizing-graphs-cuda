#include "common.cuh"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

void allocatePointCloudInt(PointCloudInt &pc, int32_t n) {
    pc.numPoints = n;
    pc.x = new int32_t[n];
    pc.y = new int32_t[n];
    pc.z = new int32_t[n];
    pc.intensity = new int32_t[n];
}

void freePointCloudInt(PointCloudInt &pc) {
    delete[] pc.x;
    delete[] pc.y;
    delete[] pc.z;
    delete[] pc.intensity;
}

void allocatePointCloudFloat(PointCloudFloat &pc, int32_t n) {
    pc.numPoints = n;
    pc.x = new float[n];
    pc.y = new float[n];
    pc.z = new float[n];
    pc.intensity = new int32_t[n];
}

void freePointCloudFloat(PointCloudFloat &pc) {
    delete[] pc.x;
    delete[] pc.y;
    delete[] pc.z;
    delete[] pc.intensity;
}

bool readInputFileInt(const std::string &filename, PointCloudInt &pc, int32_t &k, int32_t &T) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open input file: " << filename << "\n";
        return false;
    }

    int32_t n;
    if (!(inFile >> n >> k >> T)) {
        std::cerr << "Error reading header variables (n, k, T) from file.\n";
        return false;
    }

    allocatePointCloudInt(pc, n);

    for (int32_t i = 0; i < n; ++i) {
        inFile >> pc.x[i] >> pc.y[i] >> pc.z[i] >> pc.intensity[i];
    }

    inFile.close();
    return true;
}

bool readInputFileFloat(const std::string &filename, PointCloudFloat &pc, int32_t &k, int32_t &T) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open input file: " << filename << "\n";
        return false;
    }

    int32_t n;
    if (!(inFile >> n >> k >> T)) {
        std::cerr << "Error reading header variables (n, k, T) from file.\n";
        return false;
    }

    allocatePointCloudFloat(pc, n);

    for (int32_t i = 0; i < n; ++i) {
        inFile >> pc.x[i] >> pc.y[i] >> pc.z[i] >> pc.intensity[i];
    }

    inFile.close();
    return true;
}

bool writeOutputFileInt(const std::string &filename, const PointCloudInt &pc,
                        const int32_t *newIntensities) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << "\n";
        return false;
    }

    for (int32_t i = 0; i < pc.numPoints; ++i) {
        outFile << pc.x[i] << " " << pc.y[i] << " " << pc.z[i] << " " << newIntensities[i] << "\n";
    }

    outFile.close();
    return true;
}

bool writeOutputFileFloat(const std::string &filename, const PointCloudFloat &pc,
                          const int32_t *newIntensities) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << "\n";
        return false;
    }

    for (int32_t i = 0; i < pc.numPoints; ++i) {
        outFile << pc.x[i] << " " << pc.y[i] << " " << pc.z[i] << " " << newIntensities[i] << "\n";
    }

    outFile.close();
    return true;
}

float calculateMAEInt(const int32_t *exact, const int32_t *approx, int32_t numPoints) {
    if (numPoints <= 0)
        return 0.0f;
    long long totalError = 0;
#pragma omp parallel for reduction(+ : totalError)
    for (int32_t i = 0; i < numPoints; ++i) {
        totalError += std::abs(exact[i] - approx[i]);
    }
    return static_cast<float>(totalError) / numPoints;
}

float calculateMAEFloat(const int32_t *exact, const int32_t *approx, int32_t numPoints) {
    if (numPoints <= 0)
        return 0.0f;
    long long totalError = 0;
#pragma omp parallel for reduction(+ : totalError)
    for (int32_t i = 0; i < numPoints; ++i) {
        totalError += std::abs(exact[i] - approx[i]);
    }
    return static_cast<float>(totalError) / numPoints;
}

std::string formatDuration(float ms) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    double ns = static_cast<double>(ms) * 1e6;

    if (ns < 1e3) {
        oss << ns << " ns";
    } else if (ns < 1e6) {
        oss << (ns / 1e3) << " µs";
    } else if (ns < 1e9) {
        oss << ms << " ms";
    } else {
        oss << (ms / 1e3) << " s";
    }

    return oss.str();
}