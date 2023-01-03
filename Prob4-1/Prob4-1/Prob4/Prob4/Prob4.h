#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
using namespace std;
using namespace chrono;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"

#define RANGE_MAX 0.5
#define RANGE_MIN -0.5

//Function to initialize the Matrix and Vector with random real numbers
void InitializeData(float* matrix, float* v,const int Size);
//Function to write the product if the product size is not greater than 1024
void WriteData(string FileName, float* matrix, float* v, float* p, const int nSize);
//function to compare CPU and GPU computed products
void Verification(float* vect, float* gpuVect, const int Size);


//CPU Matrix Vector Multiplication Function
void cpuMatrixVectorMult(float* matrix, float* v, float* p, const int Size);


//GPU Functions
/*Helper Function: h_P should contain the product*/
__host__ void gpuMultHelper(float* h_Matrix, float* h_V, float* h_P, const int Size);
//Kernel function: g_P should contain the Product
__global__ void MatrixVectorMult(float* g_Matrix, float* g_V, float* g_P, const int Size);

