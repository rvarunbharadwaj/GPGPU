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
void ZeroMatrix(float* temp, const int ny, const int nx);
void VerifyTranspose(float* gpuTranspose, float* cpuTranspose, int ny, int nx);
void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx);

//GPU Transpose Functions
__host__ void TransposeOnGPU(float* h_Matrix, float* h_MatrixTranspose, float *refTranspose, int ny, int nx);
//Load by Row and Store by Column
__global__ void NaiveRowTranspose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx);
//Load by Column and Store by Row
__global__ void NaiveColTranspose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx);

