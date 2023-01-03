#include "VectorAddition.h"

__host__ void MultipleStreamsWOScheduling(const int nSize)
{
	//Allocating pinned memory on the host using cudaHostAlloc
	float* h_A{}, * h_B{}, * h_C_CPU{}, * h_C_GPU{};
	HandleCUDAError(cudaHostAlloc((void**)&h_A, (SIZE * sizeof(float)), cudaHostAllocDefault));
	HandleCUDAError(cudaHostAlloc((void**)&h_B, (SIZE * sizeof(float)), cudaHostAllocDefault));
	HandleCUDAError(cudaHostAlloc((void**)&h_C_CPU, (SIZE * sizeof(float)), cudaHostAllocDefault));
	HandleCUDAError(cudaHostAlloc((void**)&h_C_GPU, (SIZE * sizeof(float)), cudaHostAllocDefault));

	//Initialize Vectors
	InitializeVector(h_A, SIZE);
	InitializeVector(h_B, SIZE);
	//Perform Addition on CPU using Pinned Memory
	CPUVectorAddition("Pinned",h_A, h_B, h_C_CPU, SIZE);

	//Device Allocations
	float ElapsedTime{};
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));

	//Declare a count variable of the device memory required
	unsigned int dev_count = 6;
	//Declare a vector of pointers
	float** dev{}; //Pointer to poniter
	//Allocate memory on the GPU using cudaMalloc function for the required device memory 
	dev = new float* [dev_count];//Vector of pointers
	for (int i = 0; i < dev_count; i++)
	{
		HandleCUDAError(cudaMalloc((void**)&dev[i], PARTITION_SIZE)); //Allocating 6 different memory segments of each PARTITION_SIZE
	}
	//Create an Explicit Multiple Streams
	cudaStream_t stream_1, stream_2; //Create stream objects
	HandleCUDAError(cudaStreamCreate(&stream_1));
	HandleCUDAError(cudaStreamCreate(&stream_2));
	//Creating number of threads/block and number of blocks
	int threads_PER_BLOCK = 256;
	int blocks_PER_GRID = (int)ceil(N / threads_PER_BLOCK);
	cout << "Vector Size = " << SIZE << endl;
	cout << "Number of Threads/Block for N = " << N << ", " << threads_PER_BLOCK << endl;
	cout << "Number of Blocks/Grid for N = " << N << ", " << blocks_PER_GRID << endl;

	//Launch mutliple Streams to perform addition of the vectors overlapped without scheduling
	HandleCUDAError(cudaEventRecord(start, 0));
	//Looping over full data using multiple streams
	for (unsigned int i = 0; i < SIZE; i += N*2)
	{
		//Copy the page locked or pinned memory of size N to the device memories for stream1 
		HandleCUDAError(cudaMemcpyAsync(dev[0], (h_A + i), PARTITION_SIZE, cudaMemcpyHostToDevice, stream_1));
		HandleCUDAError(cudaMemcpyAsync(dev[1], (h_B + i), PARTITION_SIZE, cudaMemcpyHostToDevice, stream_1));
		//Launch the kernel to perform addition on the size N of the vectors on stream 1
		AddVectors << <blocks_PER_GRID, threads_PER_BLOCK, 0, stream_1 >> > (dev[0], dev[1], dev[2], N);

		//Copy the the result device memory to the host pinned memory using stream1 
		HandleCUDAError(cudaMemcpyAsync((h_C_GPU + i), dev[2], PARTITION_SIZE, cudaMemcpyDeviceToHost, stream_1));

		//Copy the page locked or pinned memory of the next size N to the device memories using stream2 
		HandleCUDAError(cudaMemcpyAsync(dev[3], (h_A + i + N), PARTITION_SIZE, cudaMemcpyHostToDevice, stream_2));
		HandleCUDAError(cudaMemcpyAsync(dev[4], (h_B + i + N), PARTITION_SIZE, cudaMemcpyHostToDevice, stream_2));
		//Launch the kernel to perform addition on the next size N of the vectors on stream 2
		AddVectors << <blocks_PER_GRID, threads_PER_BLOCK, 0, stream_2 >> > (dev[3], dev[4], dev[5], N);
		//Copy the the result device memory to the host pinned memory using stream2 
		HandleCUDAError(cudaMemcpyAsync((h_C_GPU + i + N), dev[5], PARTITION_SIZE, cudaMemcpyDeviceToHost, stream_2));
	}
	HandleCUDAError(cudaStreamSynchronize(stream_1));
	HandleCUDAError(cudaStreamSynchronize(stream_2));
	HandleCUDAError(cudaEventRecord(stop, 0));
	HandleCUDAError(cudaEventSynchronize(stop));
	HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop));
	cout << "GPU Execution Stream Version: " << ElapsedTime << " msecs" << endl;
	VerifyResults(h_C_CPU, h_C_GPU, SIZE);

	//Release the pinned memory on the host allocated
	HandleCUDAError(cudaFreeHost(h_A));
	HandleCUDAError(cudaFreeHost(h_B));
	HandleCUDAError(cudaFreeHost(h_C_CPU));
	HandleCUDAError(cudaFreeHost(h_C_GPU));

	//Destroy the stream objects
	HandleCUDAError(cudaStreamDestroy(stream_1));
	HandleCUDAError(cudaStreamDestroy(stream_2));
	//Release the allocated memory on the device
	for (int i = 0; i < dev_count; i++)
	{
		HandleCUDAError(cudaFree(dev[i]));
	}
	//Release the memory allocated on the host for the vector of pointers
	delete[] dev;

	//Destroy the event object
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));

	//Reset the device before exiting for profiler tools like Nsight and Visual Profiler to show complete traces
	HandleCUDAError(cudaDeviceReset());
}