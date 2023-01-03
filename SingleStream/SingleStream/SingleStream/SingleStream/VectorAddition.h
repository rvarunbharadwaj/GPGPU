#pragma once
#include <iostream>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

#define RANGE_MAX 1.0
#define RANGE_MIN -1.0

#define N 1024*1024
#define SIZE N*20	//Desired Vector Size
#define PARTITION_SIZE N*sizeof(float)

#define VECTOR_SIZE_IN_BYTES (SIZE*sizeof(float)) //Vector size in Bytes

void InitializeVector(float* vect, const int nSize);
void DisplayVector(string name, float* vect, const int nSize);
void VerifyResults(float* vectRef, float* vectGPU, const int nSize);

//CPU Vector Addition 
void CPUVectorAddition(float* vectA, float* vectB, float* vectC, const int nSize);

//GPU Functions
//A functions that contains the code that needs to be executed on the CPU but makes calls to CUDART library functions is known as a host/helper function. The host/helper function definition or body has to be in a .cu file for both the C and nvcc compilers to compile and should be defined with the CUDA keyword __host__
__host__ void WithDefaultStream(float* h_A, float* h_B, float* h_C_GPU, const int nSize);
__host__ void WithExplicitStream(const int nSize);

//GPU kernel or device function has to be defined with keyword __global__ and it called or launched by the host function

__global__ void AddVectors(float* g_A, float* g_B, float* g_C, int Size);
