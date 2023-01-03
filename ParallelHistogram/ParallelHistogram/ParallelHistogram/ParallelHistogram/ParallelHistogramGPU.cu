#include "ParallelHistogram.h"
//Kernel Version 0
__global__ void gpu_Histogram(unsigned char* in, int* out, unsigned int h, unsigned int w,unsigned int SIZE)
{
	unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx < (w * h))
	{
		int temp = in[idx]; //Grayscale value of a pixel
		//out[temp]++;
		atomicAdd(&(out[temp]), 1);
	}
}
#define LOCAL_BINS 256
__global__ void gpu_HistogramVer2(unsigned char* in, int* out, unsigned int h, unsigned int w, unsigned int SIZE)
{
	unsigned int global_idx = threadIdx.x + (blockIdx.x * blockDim.x); //global index is used to access global histogram
	unsigned int local_idx = threadIdx.x;
	__shared__ int local_histogram[LOCAL_BINS];
	for (int i = local_idx; i < LOCAL_BINS; i += blockDim.x) //generalization
	{
		local_histogram[i] = 0;
	}
	__syncthreads();
	if (global_idx < (w * h))
	{
		int temp = in[global_idx];
		atomicAdd(&(local_histogram[temp]), 1);
	}
	__syncthreads(); //Ready to merge after this
	for (int i = local_idx; i < LOCAL_BINS; i += blockDim.x)
	{
		atomicAdd(&(out[i]), local_histogram[i]); 
	}
}

__global__ void gpu_SharedWithThreadCoarse(unsigned char* in, int* out, unsigned int h, unsigned int w, unsigned int SIZE, unsigned int CFACTOR)
{
	unsigned int global_idx = threadIdx.x + (blockIdx.x * blockDim.x); //global index is used to access global histogram
	unsigned int local_idx = threadIdx.x;
	__shared__ int local_histogram[LOCAL_BINS];
	for (int i = local_idx; i < LOCAL_BINS; i += blockDim.x) //generalization
	{
		local_histogram[i] = 0;
	}
	__syncthreads();
	if (global_idx < (w * h))
	{
		//Histogram construction with thread corsening using contigious partitioning if CFATOR = 1 its fine granularity
		for (unsigned int i = global_idx * CFACTOR; i < min((global_idx + 1) * CFACTOR, (h * w)); ++i)
		{
			int temp = in[i];
			atomicAdd(&(local_histogram[temp]), 1);
		}
	}
	__syncthreads();
	//Ready to commit
	for (int i = local_idx; i < LOCAL_BINS; i += blockDim.x)
	{
		atomicAdd(&(out[i]), local_histogram[i]);
	}
}

__global__ void gpu_SharedWithThreadCoarseAggregation(unsigned char* in, int* out, unsigned int h, unsigned int w, unsigned int SIZE, unsigned int CFACTOR)
{
	unsigned int global_idx = threadIdx.x + (blockIdx.x * blockDim.x); //global index is used to access global histogram
	unsigned int local_idx = threadIdx.x;
	__shared__ int local_histogram[LOCAL_BINS];
	for (int i = local_idx; i < LOCAL_BINS; i += blockDim.x) //generalization
	{
		local_histogram[i] = 0;
	}
	__syncthreads();
	//Define a register variable to accumulate grayscale values
	unsigned int accumulator = 0u;
	//Define a register varibale to track the previous bin index
	int prevBinIndex = -1;
	if (global_idx < (w * h))
	{
		//Histogram construction with thread corsening using contigious partitioning if CFATOR = 1 its fine granularity
		for (unsigned int i = global_idx * CFACTOR; i < min((global_idx + 1) * CFACTOR, (h * w)); ++i)
		{
			int bin = in[i];
			if (bin == prevBinIndex)
			{
				++accumulator; //registers are local to threads, so no need to do atomic add
			}
			else
			{
				if (accumulator > 0) //its a  register so it takes 1 cycle even if we use if
				{
					atomicAdd(&(local_histogram[prevBinIndex]), accumulator);
				}
				accumulator = 1;
				prevBinIndex = bin;
			}
		}
		if (accumulator > 0)
		{
			atomicAdd(&(local_histogram[prevBinIndex]), accumulator);
		}
	}
	__syncthreads();
	//Ready to commit
	for (int i = local_idx; i < LOCAL_BINS; i += blockDim.x)
	{
		atomicAdd(&(out[i]), local_histogram[i]);
	}
}

//Host Helper function
__host__ void gpu_HistogramHelper(unsigned char* h_in, 
	unsigned int* histogram,
	unsigned int graySIZE,
	unsigned int h,
	unsigned int w,
	unsigned int BinSize,
	unsigned int* cpu_hist)
{
	unsigned char* d_in;
	int *d_out;
	//Allocating device memory for GrayScale Image and Histogram
	if (!HandleCUDAError(cudaMalloc((void**)&d_in, graySIZE)))
	{
		cout << "Error Allocating memory on GPU for the GrayScale image" << endl;
		return;
	}
	if (!HandleCUDAError(cudaMalloc((void**)&d_out, BinSize*sizeof(int))))
	{
		cout << "Error Allocating memory on GPU for the Histogram" << endl;
		return;
	}
	//Copying the GrayScale image to the device
	if (!HandleCUDAError(cudaMemcpy(d_in,h_in,graySIZE,cudaMemcpyHostToDevice)))
	{
		cout << "Error copying Gray Scale image from Host to GPU" << endl;
		return;
	}
	//Initialize the device memory for the histogram with zero
	if (!HandleCUDAError(cudaMemset(d_out,0,BinSize*sizeof(int))))
	{
		cout << "Error initializing the histogram device memory on the  GPU" << endl;
		return;
	}
	//Setup Execution Configuration Parameters
	unsigned int threadsPerBlock = 256;
	unsigned int blocksPerGrid = ((w * h) / threadsPerBlock) + 1;
	cout << "Image Grid Size = " << (w * h) << " pixels" << endl;
	cout << "Number of threads per block = " << threadsPerBlock << endl;
	cout << "Number of blocks per Grid = " << blocksPerGrid << endl;
	cout << "Total Number of Threads in the Grid = " << threadsPerBlock * blocksPerGrid << endl;
	cout << "Executing Ver 1" << endl;
	gpu_Histogram << <blocksPerGrid, threadsPerBlock >> > (d_in,
		d_out,
		h,
		w,
		BinSize);
	cudaDeviceSynchronize();
	if (!HandleCUDAError(cudaMemcpy(histogram, d_out, (BinSize*sizeof(int)), cudaMemcpyDeviceToHost)))
	{
		cout << "Error copying Histogram from GPU to Host" << endl;
		return;
	}
	Verify(cpu_hist, histogram, BinSize);
	WriteHistograms("testVer1.csv", cpu_hist,histogram, BinSize);
	//Initialize the device memory for the histogram with zero
	if (!HandleCUDAError(cudaMemset(d_out, 0, BinSize * sizeof(int))))
	{
		cout << "Error initializing the histogram device memory on the  GPU" << endl;
		return;
	}
	cout << "Executing Ver 2" << endl;
	gpu_HistogramVer2 << <blocksPerGrid, threadsPerBlock >> > (d_in,
		d_out,
		h,
		w,
		BinSize);
	cudaDeviceSynchronize();
	if (!HandleCUDAError(cudaMemcpy(histogram, d_out, (BinSize * sizeof(int)), cudaMemcpyDeviceToHost)))
	{
		cout << "Error copying Histogram from GPU to Host" << endl;
		return;
	}
	Verify(cpu_hist, histogram, BinSize);
	WriteHistograms("testVer2.csv", cpu_hist, histogram, BinSize);
	//Histogram construction with Thread coarsening
	cout << "Executing the Thread Coarsening Kernel" << endl;
	for (unsigned int i = 0; i < BinSize; i++)
	{
		histogram[i] = 0;
	}
	if (!HandleCUDAError(cudaMemset(d_out, 0, BinSize * sizeof(int))))
	{
		cout << "Error initializing the histogram device memory on the  GPU" << endl;
		return;
	}
	//Define a variable for coarsening factor
	unsigned int coarse_factor = 32u; //Explicitly mentioning its unsigned with "u"
	blocksPerGrid = ((w * h) / (threadsPerBlock * coarse_factor)) + 1; //coarse factor determines blockspergrid
	cout << "Coarse Factor" << coarse_factor << "\tBlocks = " << blocksPerGrid << endl;
	//Call the thread coarsening Kernel
	gpu_SharedWithThreadCoarse << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, h, w, BinSize, coarse_factor);
	cudaDeviceSynchronize();
	if (!HandleCUDAError(cudaMemcpy(histogram, d_out, (BinSize * sizeof(int)), cudaMemcpyDeviceToHost)))
	{
		cout << "Error copying Histogram from GPU to Host" << endl;
		return;
	}
	Verify(cpu_hist, histogram, BinSize);
	WriteHistograms("ThreadCoarse.csv", cpu_hist, histogram, BinSize);

	//Histogram construction with Thread coarsening Aggregation
	cout << "Executing the Thread Coarsening with Aggregation Kernel" << endl;
	for (unsigned int i = 0; i < BinSize; i++)
	{
		histogram[i] = 0;
	}
	if (!HandleCUDAError(cudaMemset(d_out, 0, BinSize * sizeof(int))))
	{
		cout << "Error initializing the histogram device memory on the  GPU" << endl;
		return;
	}
	//Define a variable for coarsening factor
	//unsigned int coarse_factor = 32u; //Explicitly mentioning its unsigned with "u"
	blocksPerGrid = ((w * h) / (threadsPerBlock * coarse_factor)) + 1; //coarse factor determines blockspergrid
	cout << "Coarse Factor" << coarse_factor << "\tBlocks = " << blocksPerGrid << endl;
	//Call the thread coarsening Kernel
	gpu_SharedWithThreadCoarseAggregation << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, h, w, BinSize, coarse_factor);
	cudaDeviceSynchronize();
	if (!HandleCUDAError(cudaMemcpy(histogram, d_out, (BinSize * sizeof(int)), cudaMemcpyDeviceToHost)))
	{
		cout << "Error copying Histogram from GPU to Host" << endl;
		return;
	}
	Verify(cpu_hist, histogram, BinSize);
	WriteHistograms("ThreadCoarseAgg.csv", cpu_hist, histogram, BinSize);

	if (!HandleCUDAError(cudaFree(d_in)))
	{
		cout << "Error freeing RGB image memory" << endl;
		return;
	}
	if (!HandleCUDAError(cudaFree(d_out)))
	{
		cout << "Error freeing GrayScale image memory" << endl;
		return;
	}
	HandleCUDAError(cudaDeviceReset());
}