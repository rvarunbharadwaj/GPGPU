#pragma once
#include <iostream>
#include <chrono>
#include <random>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

#define N 1024*1024
#define SIZE N*20	//Desired Vector Size
#define PARTITION_SIZE N*sizeof(float)

#define VECTOR_SIZE_IN_BYTES (SIZE*sizeof(float)) //Vector size in Bytes

void InitializeVector(float* vect, const int nSize);
void DisplayVector(string name, float* vect, const int nSize);
void VerifyResults(float* vectRef, float* vectGPU, const int nSize);

//CPU Vector Addition 
void CPUVectorAddition(string memType,float* vectA, float* vectB, float* vectC, const int nSize);

//GPU Functions
__host__ void MultipleStreamsWOScheduling(const int nSize);
__host__ void MultipleStreamsWithScheduling(const int nSize);

//GPU kernel or device function has to be defined with keyword __global__ and it called or launched by the host function
__global__ void AddVectors(float* g_A, float* g_B, float* g_C, int Size);

