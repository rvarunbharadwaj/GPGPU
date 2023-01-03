#include "VectorAddition.h"

//Device or kernel function
__global__ void VectorAddition(float* g_A, float* g_B, float* g_C, int Size)
//threadIdx.x will have indexes ranging from 0 to 1023
{
const int idx = threadIdx.x + (blockIdx.x * blockDim.x);//blockDim.x __ 0 to 1023
if (idx < Size)
{
	g_C[idx] = g_A[idx] + g_B[idx]; //unrolling . Each time we are fetching 12 bytes. One floating point operation
}
}

//Device Helper Functions
void GPUAdditionHelper(float* h_A, float* h_B, float* h_C_GPU, const int nSize) //will be compiled by both ansi c and cuda compiler
{
	float* dev_a{}, * dev_b{}, * dev_c{}; //pointer with curly brackets automatically initializes the pointer. Pointer should be a null pointer 
	chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double>elapsed_seconds;

	//Allocate memory on the GPU using the cudaMalloc function for the three vectors
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, VECTOR_SIZE_IN_BYTES);
	if (cudaStatus != cudaSuccess) //error detection
	{
		cout << "dev_a: cudaMalloc Failed" << endl;
		return;
	}
	cudaMalloc((void**)&dev_b, VECTOR_SIZE_IN_BYTES);
	cudaMalloc((void**)&dev_c, VECTOR_SIZE_IN_BYTES);

	//Copy data on the host to the device using the cudaMemcpy function
	cudaMemcpy(dev_a, h_A, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, h_B, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	
	//Creating number of threads/block and number of blocks
	int threads_per_block = 1024;
	int blocks_per_grid = (int)ceil(SIZE / threads_per_block);
	cout << "Vector Size = " << SIZE << endl;
	cout << "Number of threads/block = "<< threads_per_block << endl;
	cout << "Number of blocks/grid = " << blocks_per_grid << endl;

	//Launch the kernel on the GPU
	start = std::chrono::system_clock::now();
	VectorAddition << <blocks_per_grid, threads_per_block >> > (dev_a, dev_b, dev_c, nSize); //When we launch the cuda run time lauches the kernel with spewcified number of blocks and thread
	//Wait for the kernel to finish 
	cudaDeviceSynchronize();
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start; //measures system ticks
	cout << "GPU execution time = " << (elapsed_seconds.count() * 1000.0f) << "msecs" << endl;
	//Copy the result from the device (GPU) to host
	cudaMemcpy(h_C_GPU, dev_c, VECTOR_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	//Release the device memory using the cudaFree function
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	//Reset the device before exiting for profiler tools like Nsight and Visual Profiler to show complete traces
	cudaDeviceReset();
}