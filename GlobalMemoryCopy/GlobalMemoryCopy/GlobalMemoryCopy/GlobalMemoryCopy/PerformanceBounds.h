#pragma once
#include <iostream>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RANGE_MAX 1000.0
#define RANGE_MIN -1000.0

void InitializeMatrix(float* matrix, int ny, int nx);

//GPU Performance Benchmark Functions

__host__ void PerformanceBounds(float* h_Matrix, int ny, int nx);
//Copying a matrix with global memory coalesced access
__global__ void CopyRowWise(float* g_Matrix, float* g_MatrixCopy, int ny, int nx);
//Copying a matrix with global memory stride access
__global__ void CopyColWise(float* g_Matrix, float* g_MatrixCopy, int ny, int nx);


