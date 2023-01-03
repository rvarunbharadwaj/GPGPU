#include "VectorAddition.h"

//Device or kernel function
__global__ void AddVectors(float* g_A, float* g_B, float* g_C, int Size)
{
	//theardIdx.x will have numbers ranging from 0 to 1023
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < Size)
	{
		g_C[idx] = g_A[idx] + g_B[idx];
	}
}

