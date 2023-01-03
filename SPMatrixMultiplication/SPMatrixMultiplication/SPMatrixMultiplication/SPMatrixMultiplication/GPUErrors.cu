#include "GPUErrors.h"

bool HandleCUDAError(cudaError_t t)
{
	if (t != cudaSuccess)
	{
		cout << cudaGetErrorString(cudaGetLastError());
		return false;
	}
	return true;
}

bool GetCUDARunTimeError()
{
	cudaError_t t = cudaGetLastError();
	if (t != cudaSuccess)
	{
		cout << cudaGetErrorString(t) << endl;
		return false;
	}
	return true;
}

