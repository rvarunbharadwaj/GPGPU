#include "ParallelReduction.h"
#include "GPUErrors.h"

__global__ void NeighborhoodWithDivergence(float* g_Vector, float* g_PartialSum)
{
	//Save threadIdx.x on the register
	int tid = threadIdx.x;

	//Compute the global thread index
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	//Compute the local pointer to each block
	float* blockAddress = g_Vector + (blockIdx.x * blockDim.x);

	//in-place reduction in global memory
	if (idx >= VECTOR_SIZE)
	{
		return;
	}
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0) //just threadIdx.x
	{
		g_PartialSum[blockIdx.x] = blockAddress[0]; //after for loop the final answer will be in 0 location
	}

}

__global__ void NeighborhoodWithLessDivergence(float* g_Vector, float* g_PartialSum)
{
	//Save threadIdx.x on the register
	int tid = threadIdx.x;

	//Compute the global thread index
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	//Compute the local pointer to each block
	float* blockAddress = g_Vector + (blockIdx.x * blockDim.x);

	//in-place reduction in global memory
	if (idx >= VECTOR_SIZE)
	{
		return;
	}
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		int index = 2 * stride * tid;
		if (index < blockDim.x)
		{
			blockAddress[index] += blockAddress[index + stride];
		}
		__syncthreads();
	}
	if (tid == 0) //just threadIdx.x
	{
		g_PartialSum[blockIdx.x] = blockAddress[0]; //after for loop the final answer will be in 0 location
	}

}

__host__ void OnNeighborhood(float* vectorTemp)
{
	chrono::time_point<std::chrono::system_clock> start, end;

	float* d_Vector;
	float* d_PartialSum;

	float* h_PartialSum;

	//Block and Thread Parameters
	dim3 block(256);
	dim3 grid((VECTOR_SIZE + block.x - 1) / block.x, 1);
	cout << "Neighborhood Implementations" << endl;
	cout << "\tThreads/Block: " << block.x << endl;
	cout << "\tBlocks/Grid: " << grid.x << endl;

	//The partial sums of each block
	h_PartialSum = new float[grid.x];

	//Allocate memory on the GPU to store the vector and partial sums
	HandleCUDAError(cudaMalloc((void**)&d_Vector, VECTOR_SIZE_IN_BYTES));
	HandleCUDAError(cudaMalloc((void**)&d_PartialSum, (grid.x * sizeof(float))));

	//Copy the vector to the GPU from the host
	HandleCUDAError(cudaMemcpy(d_Vector, vectorTemp, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice));

	//Launch the Neighboorhood pairing kernel with Divergence
	NeighborhoodWithDivergence << <grid, block >> > (d_Vector, d_PartialSum);
	cudaDeviceSynchronize();
	
	//Copy the vector to the GPU from the host containing the sum of each block
	HandleCUDAError(cudaMemcpy(h_PartialSum, d_PartialSum, (grid.x * sizeof(float)), cudaMemcpyDeviceToHost));
	//do reduction 
	float sum = 0.0f;
	for (int j = 0; j < grid.x; j++)
	{
		sum += h_PartialSum[j];
	}
	cout << "\t\tGPU Neighborhood Reduction: " << sum << endl;

	//Release Global Memory of d_Vector, and d_PartialSum
	HandleCUDAError(cudaFree(d_Vector));
	HandleCUDAError(cudaFree(d_PartialSum));

	//Reallocate Global Memory of d_Vector, and d_PartialSum
	HandleCUDAError(cudaMalloc((void**)&d_Vector, VECTOR_SIZE_IN_BYTES));
	HandleCUDAError(cudaMalloc((void**)&d_PartialSum, (grid.x * sizeof(float))));
	//Copy the vector to the GPU from the host
	HandleCUDAError(cudaMemcpy(d_Vector, vectorTemp, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice));
	
	//Launch the Neighborhood pairing kernel with less divergence
	NeighborhoodWithLessDivergence << <grid, block >> > (d_Vector, d_PartialSum);
	cudaDeviceSynchronize();

	//Copy the vector to the GPU from the host containing the sum of each block
	HandleCUDAError(cudaMemcpy(h_PartialSum, d_PartialSum, (grid.x * sizeof(float)), cudaMemcpyDeviceToHost));
	sum = 0.0f;
	for (int j = 0; j < grid.x; j++)
	{
		sum += h_PartialSum[j];
	}
	cout << "\t\tGPU Neighborhood Reduction: " << sum << endl;

	HandleCUDAError(cudaFree(d_Vector));
	HandleCUDAError(cudaFree(d_PartialSum));
	HandleCUDAError(cudaDeviceReset());
}