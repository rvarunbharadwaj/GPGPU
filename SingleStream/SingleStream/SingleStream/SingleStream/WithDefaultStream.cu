#include "VectorAddition.h"

__host__ void WithDefaultStream(float* h_A, float* h_B, float* h_C_GPU, const int nSize)
{
	float ElapsedTime{};
	//Create GPU Event Objects
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));

	float* dev_a{}, * dev_b{}, * dev_c{};
	//Allocate memory on the GPU using the cudaMalloc function for the three vectors only for N elements
	HandleCUDAError(cudaMalloc((void**)&dev_a, PARTITION_SIZE));
	HandleCUDAError(cudaMalloc((void**)&dev_b, PARTITION_SIZE));
	HandleCUDAError(cudaMalloc((void**)&dev_c, PARTITION_SIZE));

	//Creating number of threads/block and number of blocks
	int threads_PER_BLOCK = 256;
	int blocks_PER_GRID = (int)ceil(N / threads_PER_BLOCK);
	cout << "Vector Size = " << SIZE << endl;
	cout << "Number of Threads/Block for N = "<<N<<", " << threads_PER_BLOCK << endl;
	cout << "Number of Blocks/Grid for N = " << N << ", " << blocks_PER_GRID << endl;

	//Launch the kernel on the GPU to add the vectors in sections
	//Record the event time of the kernel execution starting
	HandleCUDAError(cudaEventRecord(start, 0));
	for (unsigned int i = 0; i < SIZE; i += N)
	{
		//Copy only a section of the host data (h_A and h_B) to the device corresponding to size of N
		HandleCUDAError(cudaMemcpy(dev_a, (h_A + i), PARTITION_SIZE, cudaMemcpyHostToDevice));
		HandleCUDAError(cudaMemcpy(dev_b, (h_B + i), PARTITION_SIZE, cudaMemcpyHostToDevice));
		//Launch the kernel for adding only the section of size N
		AddVectors << <blocks_PER_GRID, threads_PER_BLOCK >> > (dev_a, dev_b, dev_c, N); //dev_a will be pointing to the first element
		
		cudaDeviceSynchronize();
		//Copy only a section of the device data (dev_c) to the host corresponding to size of N
		HandleCUDAError(cudaMemcpy((h_C_GPU + i), dev_c, PARTITION_SIZE, cudaMemcpyDeviceToHost));
	}
	//Record the event time of the kernel execution completition
	HandleCUDAError(cudaEventRecord(stop, 0));
	//Block the host to receive a synchronization recording event from the GPU
	HandleCUDAError(cudaEventSynchronize(stop));

	//Compute the kernel execution time
	HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop));
	cout << "GPU Execution Non Stream Version: " << ElapsedTime << " msecs" << endl;

	//Release the allocated memory on the device
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	//Destroy the CUDA Event objects
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));

	//Reset the device before exiting for profiler tools like Nsight and Visual Profiler to show complete traces
	HandleCUDAError(cudaDeviceReset());
}