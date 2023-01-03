#include "Prob4.h"
#include "GPUErrors.h"

__global__ void MatrixVectorMult(float* g_Matrix, float* g_V, float* g_P, const int Size)
{
	//Write code to perform Matrix Vector Multiplication
	int idx = threadIdx.x + (blockIdx.x * blockDim.x); //Calculating thread index
	float sum; //Register to store summation
	if (idx < Size)
	{
		sum = 0.0f; //Initialize to zero every iteration
		for (int k = 0; k < Size; k++)
		{
			sum += g_V[k] * g_Matrix[idx * Size + k];
		}
		g_P[idx] = sum; //Storing the value in output Vector
	}
}

__host__ void gpuMultHelper(float* h_Matrix, float* h_V, float* h_P, const int Size)
{
	float* d_Matrix, * d_V, * d_P;
	const int MatrixSizeInBytes = Size * Size * sizeof(float); //Size of Matrix in Bytes
	const int VectorSize = Size * sizeof(float); //Size of Vector in Bytes

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_Matrix, MatrixSizeInBytes)); //Allocate memory for d_Matrix
	HandleCUDAError(cudaMalloc((void**)&d_V, VectorSize)); //Allocate memory for d_V
	HandleCUDAError(cudaMalloc((void**)&d_P, VectorSize)); //Allocate memory for d_P

	//Transfer data from CPU Memory to GPU Memory and handle any errors
	HandleCUDAError(cudaMemcpy(d_Matrix, h_Matrix, MatrixSizeInBytes, cudaMemcpyHostToDevice)); //Copying the matrix from host to device
	HandleCUDAError(cudaMemcpy(d_V, h_V, VectorSize, cudaMemcpyHostToDevice)); //Copying the Vector from host to device

	//Kernel Execution Configuration Parameters 
	int threads_per_block = 256; //Threads per BLock
	int blocks_per_grid = (int)ceil(Size / threads_per_block); //Blocks per grid

	//Launch Kernel and collect execution time
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};
	start = high_resolution_clock::now();
	MatrixVectorMult << <blocks_per_grid, threads_per_block >> > (d_Matrix, d_V, d_P, Size); //Kernel Parameter Passing
	//cudaDeviceSynchronize(); //Make the CPU wait

	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "GPU Execution time: " << computeTime << " usecs" << endl;

	//Transfer data from GPU Memory to CPU memory and handle any errors
	HandleCUDAError(cudaMemcpy(h_P, d_P, VectorSize, cudaMemcpyDeviceToHost)); //Transfer the value of Output Vector from device to host

	//Release device memory
	HandleCUDAError(cudaFree(d_Matrix)); //Release the memory allocated for Matrix
	HandleCUDAError(cudaFree(d_V)); //Release the memory allocated for Vector
	HandleCUDAError(cudaFree(d_P)); //Release the memory allocated for Output Vector

	HandleCUDAError(cudaDeviceReset()); //Explicitly destroy and clean up all resources associated with the current device in the current process
}