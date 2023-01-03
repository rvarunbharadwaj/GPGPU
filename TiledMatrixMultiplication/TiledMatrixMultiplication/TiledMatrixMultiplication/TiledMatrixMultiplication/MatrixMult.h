#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

void InitializeMatrix(float* matrix, int ny, int nx);
void ZeroMatrix(float* temp, const int ny, const int nx);
void MatrixMultVerification(float* hostC, float* gpuC, const int ny, const int nx);
void DisplayMatrix(string name, float* temp, const int ny, const int nx);

//CPU Implementations
void cpuMatrixMult(float* A, float* B, float* C, const int ny, const int nx);

//GPU Functions
__host__ void gpuMultHelper(float* h_A, float* h_B, float* h_C, float* h_C_Tile, float* ref, const int ny, const int nx);
__global__ void NaiveMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx);
//TiledMult kernel works only for Square Matrices
__global__ void TiledMult(float* g_A, float* g_B, float* g_C, const int Width);

