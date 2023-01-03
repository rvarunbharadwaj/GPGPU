#pragma once
#include <iostream>
#include <chrono>
#include <random>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

#define N 1024
#define SIZE N*127	//Desired Vector Size


#define VECTOR_SIZE_IN_BYTES (SIZE*sizeof(int)) //Vector size in Bytes

void InitializeVector(int* vect, const int nSize);

//GPU helper Function to count odds using shared memory kernel
__host__ void CountOddsHelper(int *h,int oc_check,const int size);
//GPU kernel to count odds using shared memory
__global__ void CountOdds(int* g_Vect, int* g_Odds, const int Size);

//GPU helper Function to count odds using shared memory with warp primitives kernel
__host__ void CountOddsWPHelper(int* h, int oc_check, const int size);
//GPU kernel to count odds using shared memory
__global__ void CountOdds_WP(int* g_Vect, int* g_Odds, const int Size);

