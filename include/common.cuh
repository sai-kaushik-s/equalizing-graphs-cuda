#pragma once
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <streambuf>
#include <fstream>
#include <memory>

class LoggerStreamBuf : public std::streambuf {
public:
    LoggerStreamBuf(std::streambuf* sb1, std::streambuf* sb2);
protected:
    virtual int_type overflow(int_type c) override;
    virtual int sync() override;
private:
    std::streambuf* sb1;
    std::streambuf* sb2;
};

class Logger {
public:
    Logger(const std::string& filename);
    ~Logger();
private:
    std::ofstream fileStream;
    std::unique_ptr<LoggerStreamBuf> teeBuf;
    std::streambuf* oldCoutBuf;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA API Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Kernel Launch Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Kernel Sync Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct PointCloud {
    float* x;
    float* y;
    float* z;
    int* intensity;
    int numPoints;
};

void allocatePointCloud(PointCloud& pc, int n);
void freePointCloud(PointCloud& pc);
bool readInputFile(const std::string& filename, PointCloud& pc, int& k, int& T);
bool writeOutputFile(const std::string& filename, const PointCloud& pc, const int* newIntensities);
std::string formatDuration(std::chrono::nanoseconds d);
void validateOutputs(const int* cpuOutput, const int* gpuOutput, int n);

template <typename Func, typename... Args>
std::chrono::nanoseconds timeFunction(const std::string& label, Func&& func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<Func>(func)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::nanoseconds duration = end - start;
    std::cout << label << ": " << formatDuration(duration) << "\n";

    return duration;
}