#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

#define S 2048
#define VECTOR_SIZE S //Number of Elements
#define VECTOR_SIZE_IN_BYTES (VECTOR_SIZE*sizeof(float))

//CPU Functions
void OnInitializeInputData(float* vectorTemp, int SIZE);
void CopyInputData(float* vectorTemp, float* ref,int SIZE);
void ZeroData(float* in, int SIZE);
void OnSequentialScan(float* in, float * out,int SIZE);
void PrintVectors(float* vector,int SIZE);
void Verify(float* ref, float* in, int SIZE);

//GPU Helper
__host__ void Helper_Scan(float* Input, float* Output, float* RefOutputData, int SIZE);

//GPU Kernel
__global__ void ScanInEfficient(float* In, float* Out,const int SIZE);
__global__ void ScanEfficient(float* In, float* Out, const int SIZE);




