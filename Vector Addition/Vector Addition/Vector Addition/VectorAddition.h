#pragma once
#include <iostream>
#include <chrono> //Chrono is the header file which contains high performance clock(present on the mother board) This is accessing the wall clock
using namespace std;
#include <cuda.h> //Contains all basic cuda api(application programming interface)
#include <cuda_runtime.h> // Contains info about how to launch the kernel and architecture
#include <device_launch_parameters.h> //Contains specific info about GPU

#define RANGE_MAX 1.0
#define RANGE_MIN -1.0

#define N 100*1024
#define SIZE 1024*N*2	//Desired Vector Size

#define VECTOR_SIZE_IN_BYTES (SIZE*sizeof(float)) //Vector size in Bytes
//*Single precision number requires 4 bytes

void InitializeVector(float* vect, const int nSize);
void DisplayVector(string name, float* vect, const int nSize); //For Debugging
void VerifyResults(float* vectRef, float* vectGPU, const int nSize);

//CPU Vector Addition 
void CPUVectorAddition(float* vectA, float* vectB, float* vectC, const int nSize);

//GPU Functions
//A functions that contains the code that needs to be executed on the CPU but makes calls to CUDART library functions is known as a host/helper function. The host/helper function definition or body has to be in a .cu file for both the C and nvcc compilers to compile and should be defined with the CUDA keyword __host__
__host__ void GPUAdditionHelper(float* h_A, float* h_B, float* h_C_GPU, const int nSize1);

//GPU kernel or device function has to be defined with keyword __global__ and it called or launched by the host function
__global__ void VectorAddition(float* g_A, float* g_B, float* g_C, int Size);

