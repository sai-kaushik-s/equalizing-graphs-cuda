#pragma once
#include <vector>
#include <string>

struct PointCloud {
    float* x;
    float* y;
    float* z;
    int* intensity;
    int num_points;
};

void allocate_point_cloud(PointCloud& pc, int n);
void free_point_cloud(PointCloud& pc);
bool read_input_file(const std::string& filename, PointCloud& pc, int& k, int& T);
bool write_output_file(const std::string& filename, const PointCloud& pc, const int* new_intensities);