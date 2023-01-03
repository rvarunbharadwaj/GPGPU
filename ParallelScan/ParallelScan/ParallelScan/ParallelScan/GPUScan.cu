#include "ParallelScan.h"

#define BLOCK_SIZE 1024
//A Kernel that scan a maximum of 1024 elements
__global__ void ScanInEfficient(float* In, float* Out, const int SIZE)
{
	int tid = threadIdx.x;
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	__shared__ float In_Shared[BLOCK_SIZE];
	if (idx < SIZE)
	{
		In_Shared[tid] = In[idx];
	}
	__syncthreads();
	for (unsigned int stride = 1; stride <= tid; stride *= 2)
	{
		__syncthreads();
		In_Shared[tid] += In_Shared[tid - stride]; //for tid =1 stride will be 1, we compute x0 + x1 //for tid = 2, we compute x1+x2
	}
	Out[idx] = In_Shared[tid];
}

__global__ void ScanEfficient(float* In, float* Out, const int SIZE)
{
	int tid = threadIdx.x;
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	__shared__ float In_Shared[BLOCK_SIZE];
	if (idx < SIZE)
	{
		In_Shared[tid] = In[idx];
	}
	else
	{
		In_Shared[tid] = 0.0f;
	}
	for (int stride = 1; stride < blockDim.x; stride *= 2)//reduction phase loop
	{
		__syncthreads();
		int index = ((tid + 1) * 2 * stride) - 1;//will be a odd index
		if (index < blockDim.x)
		{
			In_Shared[index] += In_Shared[index - stride];
		}
	}
	for (int stride = (BLOCK_SIZE / 4); stride > 0; stride /= 2)//Reverse phase
	{
		__syncthreads();
		int index = ((tid + 1) * 2 * stride) - 1;
		if ((index + stride) < blockDim.x)
		{
			In_Shared[index + stride] += In_Shared[index];
		}
	}
	__syncthreads();
	if (idx < SIZE)
	{
		Out[idx] = In_Shared[tid];
	}
}

__host__ void Helper_Scan(float* Input, float* Output, float* RefOutputData, int SIZE)
{
	float* d_in;
	float* d_out;

	//Allocate memory on the GPU to store the vector and scanned output
	HandleCUDAError(cudaMalloc((void**)&d_in, VECTOR_SIZE_IN_BYTES));
	HandleCUDAError(cudaMalloc((void**)&d_out, VECTOR_SIZE_IN_BYTES));

	//Copy the vector to the GPU from the host
	HandleCUDAError(cudaMemcpy(d_in, Input, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice));


	//Launch Inefficient Kernel
	ScanInEfficient <<<2,BLOCK_SIZE>>> (d_in, d_out, SIZE);
	cudaDeviceSynchronize();
	//Copy the vector to the GPU from the host containing the sum of each block
	HandleCUDAError(cudaMemcpy(Output, d_out, VECTOR_SIZE_IN_BYTES, cudaMemcpyDeviceToHost));
	Verify(RefOutputData, Output, SIZE);
	cout << "Inefficient GPU Results" << endl;
	PrintVectors(Output, SIZE);

	cudaMemset(d_out, 0.0, VECTOR_SIZE_IN_BYTES);

	ScanEfficient << <2, BLOCK_SIZE >> > (d_in, d_out, SIZE);
	cudaDeviceSynchronize();
	//Copy the vector to the GPU from the host containing the sum of each block
	HandleCUDAError(cudaMemcpy(Output, d_out, VECTOR_SIZE_IN_BYTES, cudaMemcpyDeviceToHost));
	Verify(RefOutputData, Output, SIZE);
	cout << "Efficient GPU Results" << endl;
	PrintVectors(Output, SIZE);

	HandleCUDAError(cudaFree(d_in));
	HandleCUDAError(cudaFree(d_out));
	HandleCUDAError(cudaDeviceReset());
}