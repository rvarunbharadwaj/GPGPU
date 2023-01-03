#include "WarpFunc.h"

__global__ void CountOdds(int* g_Vect, int* g_Odds, const int Size)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int totalThreads = blockDim.x * gridDim.x;
	int localID = threadIdx.x;

	//Dynamic Shared Memory Allocation //Dynamic Shared memory can be only done in 1D
	extern __shared__ int count[]; //Vector of counters with size Block_sized

	//Using Thread Coarsening to have maximum active blocks per SM
	for (int i = idx; i < Size; i += totalThreads)
	{
		count[localID] += (g_Vect[i] % 2); //Size of the count should be block_size
	}
	//count[localID] += (g_Vect[idx] % 2); //Size of the count should be block_size
	__syncthreads();

	//Reduction phase: Summing up the block
	int step = 1;
	int otherIdx = localID | step; //Perform a bitwise OR
	while ((otherIdx < blockDim.x) && ((localID & step) == 0))
	{
		count[localID] += count[otherIdx];
		step <<= 1;
		otherIdx = localID | step;
		__syncthreads();
	}
	//Add to the global counts
	if (localID == 0)
	{
		atomicAdd(g_Odds, count[0]); //We'll use atomic add as many as blocks
	}
	
}

__host__ void CountOddsHelper(int* h, int oc_check, const int size)
{
	//Variable to store the GPU computed odd counts;
	int gpu_OddCount;
	//Device pointer for the input vector on the Global Memory
	int* d_vect{};
	//Device pointer for the odd count result from the kernel
	int* d_oc{};
	//Allocate Global Memory for the input vector and the odd count
	HandleCUDAError(cudaMalloc((void**)&d_vect, VECTOR_SIZE_IN_BYTES));
	HandleCUDAError(cudaMalloc((void**)&d_oc, sizeof(int) * 1)); //Just to store one value

	//Copy input data to the device memory
	HandleCUDAError(cudaMemcpy(d_vect, h, VECTOR_SIZE_IN_BYTES, cudaMemcpyHostToDevice));
	//Initialize the device odd count memory to zero
	HandleCUDAError(cudaMemset(d_oc, 0, sizeof(int)*1));

	//Determining execution configuration for maximum active Blocks
	//Determine the number of SMs on the GPU
	cudaDeviceProp prop;
	int device_id;
	HandleCUDAError(cudaGetDevice(&device_id));
	HandleCUDAError(cudaGetDeviceProperties(&prop, device_id));
	int SM = prop.multiProcessorCount;
	cout <<endl<< "Number of SMs: " << SM << endl;

	int blockSize = 256;
	int blockPerSM{}, gridSize{};
	//Determine the shared memory size for dynamic shared memory allocation
	int SharedMemSize = blockSize * sizeof(int); //1KB size cause 256 * 4 bytes
	//Call the execution configuration function cudaOccupancyMaxActiveBlocksPerMultiprocessor API to determine blocks per SM
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockPerSM, (void*)CountOdds, blockSize, SharedMemSize);
	//Compute grid size
	gridSize = min((int)ceil(1.0 * SIZE / blockSize), blockPerSM * SM);
	//Display Execution Configuration
	cout << "Number of Threads per Block: " << blockSize << endl;
	cout << "Number of suggested blocks per SM for Maximum Active Blocks Per SM: " << blockPerSM << endl;
	cout << "Number of Blocks per Grid: " << gridSize << endl;

	CountOdds << <gridSize, blockSize, SharedMemSize >> > (d_vect, d_oc, size);
	cudaDeviceSynchronize();
	HandleCUDAError(cudaMemcpy(&gpu_OddCount, d_oc, sizeof(int)*1, cudaMemcpyDeviceToHost));

	//Verify results
	if (gpu_OddCount == oc_check)
	{
		cout << "Number of Odds  determined by the GPU without Warp Primitives: " << gpu_OddCount << endl;
	}
	else {
		cout << "Number of Odds  determined by the GPU without Warp Primitives (Error):  " << gpu_OddCount << endl;
	}

	HandleCUDAError(cudaFree(d_vect));
	HandleCUDAError(cudaFree(d_oc));
	HandleCUDAError(cudaDeviceReset());
}