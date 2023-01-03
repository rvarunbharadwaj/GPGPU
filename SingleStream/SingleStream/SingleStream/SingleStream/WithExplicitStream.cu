#include "VectorAddition.h"

__host__ void WithExplicitStream(const int nSize)
{
	float ElapsedTime{};
	//Determine whether the GPU Supports Streaming i.e., allows execution overlapping of kernels
	cudaDeviceProp prop;
	int device_id;
	HandleCUDAError(cudaGetDevice(&device_id));
	HandleCUDAError(cudaGetDeviceProperties(&prop, device_id));
	cout << "Number of Asynchronous Engines: " << prop.asyncEngineCount << endl;
	if (!prop.concurrentKernels)
	{
		cout << "Device does not handle concurrent execution of kernels" << endl;
		return;
	}
	else {
		cout << "Device handles overlaps or streams" << endl; //if we copy from host to data on two streams both will be copied comcurrently. Only certain GPUs allow it.
	}

	float* h_A, * h_B, * h_C_CPU, * h_C_GPU;
	//Allocating pinned memory on the host using cudaHostAlloc
	HandleCUDAError(cudaHostAlloc((void**)&h_A, (SIZE * sizeof(float)), cudaHostAllocDefault)); //will automatically allocate data with byte boundary
	HandleCUDAError(cudaHostAlloc((void**)&h_B, (SIZE * sizeof(float)), cudaHostAllocDefault));
	HandleCUDAError(cudaHostAlloc((void**)&h_C_GPU, (SIZE * sizeof(float)), cudaHostAllocDefault));
	HandleCUDAError(cudaHostAlloc((void**)&h_C_CPU, (SIZE * sizeof(float)), cudaHostAllocDefault));

	//Initialize Vectors
	InitializeVector(h_A, SIZE);
	InitializeVector(h_B, SIZE);
	CPUVectorAddition(h_A, h_B, h_C_CPU, SIZE); //size of h_A + h_B should be less than physical RAM

	//Device Allocations

	//Create GPU Event Objects
	cudaEvent_t start, stop;
	HandleCUDAError(cudaEventCreate(&start));
	HandleCUDAError(cudaEventCreate(&stop));

	float* dev_a{}, * dev_b{}, * dev_c{};
	//Allocate memory on the GPU using the cudaMalloc function for the three vectors only for N elements
	HandleCUDAError(cudaMalloc((void**)&dev_a, PARTITION_SIZE));
	HandleCUDAError(cudaMalloc((void**)&dev_b, PARTITION_SIZE));
	HandleCUDAError(cudaMalloc((void**)&dev_c, PARTITION_SIZE));

	//Declaring a cudaStream_t variable
	cudaStream_t stream_1; //creating an object for explicit stream
	//Create an Explicit Stream Object
	HandleCUDAError(cudaStreamCreate(&stream_1));

	//Creating number of threads/block and number of blocks
	int threads_PER_BLOCK = 256;
	int blocks_PER_GRID = (int)ceil(N / threads_PER_BLOCK);
	cout << "Vector Size = " << SIZE << endl;
	cout << "Number of Threads/Block for N = " << N << ", " << threads_PER_BLOCK << endl;
	cout << "Number of Blocks/Grid for N = " << N << ", " << blocks_PER_GRID << endl;

	//Launch the kernel on the GPU to add the vectors in sections using Stream and Asynchronously
	//Record the event time of the kernel execution starting
	HandleCUDAError(cudaEventRecord(start, stream_1));
	for (unsigned int i = 0; i < SIZE; i += N)
	{
		//Copy Asynchronously only a section of the host data (h_A and h_B) to the device corresponding to size of N
		HandleCUDAError(cudaMemcpyAsync(dev_a, (h_A + i), PARTITION_SIZE, cudaMemcpyHostToDevice, stream_1)); //cudaMemcpyAsync is a non blocking call
		HandleCUDAError(cudaMemcpyAsync(dev_b, (h_B + i), PARTITION_SIZE, cudaMemcpyHostToDevice, stream_1));
		//Launch the kernel on the explicit stream // third param with value 0 is shared memory size for dynamic alloc
		AddVectors << <blocks_PER_GRID, threads_PER_BLOCK, 0, stream_1 >> > (dev_a, dev_b, dev_c, N);
		//Copy Asynchronously only a section of the device data (dev_c) to the host corresponding to size of N
		HandleCUDAError(cudaMemcpyAsync((h_C_GPU + i), dev_c, PARTITION_SIZE, cudaMemcpyHostToDevice, stream_1));
	}
	//Block the host for GPU to synchronize with completition of the stream operations
	HandleCUDAError(cudaStreamSynchronize(stream_1));

	//Record the event time of the kernel execution completition
	HandleCUDAError(cudaEventRecord(stop, stream_1));
	//Block the host to receive a synchronization event of recording from the GPU
	HandleCUDAError(cudaEventSynchronize(stop));

	HandleCUDAError(cudaEventElapsedTime(&ElapsedTime, start, stop));
	cout << "GPU Execution Stream Version: " << ElapsedTime << " msecs" << endl;
	VerifyResults(h_C_CPU, h_C_GPU, SIZE);

	//Release the pinned memory on the host allocated
	HandleCUDAError(cudaFreeHost(h_A));
	HandleCUDAError(cudaFreeHost(h_B));
	HandleCUDAError(cudaFreeHost(h_C_CPU));
	HandleCUDAError(cudaFreeHost(h_C_GPU));

	//Destroy the stream object
	HandleCUDAError(cudaStreamDestroy(stream_1));

	//Release the allocated memory on the device
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	//Destroy the event object
	HandleCUDAError(cudaEventDestroy(start));
	HandleCUDAError(cudaEventDestroy(stop));

	//Reset the device before exiting for profiler tools like Nsight and Visual Profiler to show complete traces
	HandleCUDAError(cudaDeviceReset());
}