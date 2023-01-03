#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

#define S 128
#define VECTOR_SIZE S*1024*1024 //Number of Elements
#define VECTOR_SIZE_IN_BYTES (VECTOR_SIZE*sizeof(float))

//CPU Functions
void OnInitializeInputData(float* vectorTemp, const int SIZE);
void CopyInputData(float* vectorTemp, float* ref,const int SIZE);
float OnSequentialReduce(float* vectorTemp, const int SIZE);
float OnCPURecursiveReduce(float* VectorTemp, const int SIZE);

//GPU Implementations - Neighborhood Pairing
__host__ void OnNeighborhood(float* vectorTemp);
__global__ void NeighborhoodWithDivergence(float* g_Vector, float* g_PartialSum);
__global__ void NeighborhoodWithLessDivergence(float* g_Vector, float* g_PartialSum);





