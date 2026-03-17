#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <streambuf>

#include "common.cuh"

LoggerStreamBuf::LoggerStreamBuf(std::streambuf* sb1, std::streambuf* sb2)
    : sb1(sb1), sb2(sb2) {}

std::streambuf::int_type LoggerStreamBuf::overflow(int_type c) {
    if (c == traits_type::eof()) {
        return traits_type::not_eof(c);
    } else {
        int_type r1 = sb1->sputc(c);
        int_type r2 = sb2->sputc(c);
        return (r1 == traits_type::eof() || r2 == traits_type::eof()) ? traits_type::eof() : c;
    }
}

int LoggerStreamBuf::sync() {
    int r1 = sb1->pubsync();
    int r2 = sb2->pubsync();
    return (r1 == 0 && r2 == 0) ? 0 : -1;
}

Logger::Logger(const std::string& filename) {
    fileStream.open(filename);
    if (!fileStream.is_open()) {
        std::cerr << "Failed to open log file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }
    teeBuf = std::make_unique<LoggerStreamBuf>(std::cout.rdbuf(), fileStream.rdbuf());
    oldCoutBuf = std::cout.rdbuf(teeBuf.get());
}

Logger::~Logger() {
    std::cout.rdbuf(oldCoutBuf);
    if (fileStream.is_open()) {
        fileStream.close();
    }
}

void allocatePointCloud(PointCloud& pc, int n) {
    pc.numPoints = n;
    pc.x = new float[n];
    pc.y = new float[n];
    pc.z = new float[n];
    pc.intensity = new int[n];
}

void freePointCloud(PointCloud& pc) {
    delete[] pc.x;
    delete[] pc.y;
    delete[] pc.z;
    delete[] pc.intensity;
}

bool readInputFile(const std::string& filename, PointCloud& pc, int& k, int& T) {
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open input file: " << filename << "\n";
        return false;
    }

    int n;
    if (!(inFile >> n >> k >> T)) {
        std::cerr << "Error reading header variables (n, k, T) from file.\n";
        return false;
    }

    allocatePointCloud(pc, n);

    for (int i = 0; i < n; ++i) {
        inFile >> pc.x[i] >> pc.y[i] >> pc.z[i] >> pc.intensity[i];
    }

    inFile.close();
    return true;
}

std::string formatDuration(std::chrono::nanoseconds d) {
    double seconds = std::chrono::duration<double>(d).count();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    if (seconds < 1e-6) {
        oss << std::chrono::duration<double, std::nano>(d).count() << " ns";
    }
    else if (seconds < 1e-3) {
        oss << std::chrono::duration<double, std::micro>(d).count() << " µs";
    }
    else if (seconds < 1.0) {
        oss << std::chrono::duration<double, std::milli>(d).count() << " ms";
    }
    else {
        oss << seconds << " s";
    }

    return oss.str();
}

bool writeOutputFile(const std::string& filename, const PointCloud& pc, const int* newIntensities) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << "\n";
        return false;
    }
    
    for (int i = 0; i < pc.numPoints; ++i) {
        outFile << pc.x[i] << " " << pc.y[i] << " " << pc.z[i] << " " << newIntensities[i] << "\n";
    }
    
    outFile.close();
    return true;
}

void validateOutputs(const int* cpuOutput, const int* gpuOutput, int n) {
    int mismatches = 0;
    int maxDifference = 0;

    for (int i = 0; i < n; ++i) {
        if (cpuOutput[i] != gpuOutput[i]) {
            mismatches++;
            int diff = std::abs(cpuOutput[i] - gpuOutput[i]);
            if (diff > maxDifference) {
                maxDifference = diff;
            }
            if (mismatches <= 5) {
                std::cout << "   Mismatch at index " << i 
                          << ": CPU=" << cpuOutput[i] 
                          << ", GPU=" << gpuOutput[i] << "\n";
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "Validation PASSED: CPU and GPU outputs match perfectly!\n";
    } else {
        std::cout << "Validation FAILED: " << mismatches << " mismatches found.\n";
        std::cout << "   Maximum intensity difference: " << maxDifference << "\n";
    }
}