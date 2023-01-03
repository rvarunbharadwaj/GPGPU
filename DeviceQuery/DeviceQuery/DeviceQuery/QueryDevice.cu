#include <iostream>
#include <cmath>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


int main()
{
	cudaDeviceProp prop;
	int nCountDevices{};

	cudaGetDeviceCount(&nCountDevices);
	cout << "Number of GPUs: " << nCountDevices << endl;

	for (int i = 0; i < nCountDevices; i++)
	{
		cudaGetDeviceProperties(&prop, i);

		cout << "Name = " << prop.name << endl;
		cout << "Compute Capability = " << prop.major << "." << prop.minor << endl; // To perform double precision floating point operation the Compute Capability should be 1.3 or higher.
		cout << "Streaming Multiprocessors (SMs) = " << prop.multiProcessorCount << endl;
		cout << "Streaming Processors/SM (SPs) =" << _ConvertSMVer2Cores(prop.major, prop.minor) << endl;
		cout << "GPU Clock Rate = " << (prop.clockRate / pow(10.0, 3.0)) << " MHz" << endl; //Clock rate is returned in kilohertz.
		cout << "Memory Clock Rate = " << (prop.memoryClockRate / pow(10.0, 3.0)) << " MHz" << endl;
		cout << "Memory Bus Width = " << prop.memoryBusWidth << " bits" << endl;

		if (prop.l2CacheSize)
		{
			cout << "L2 Cache size = " << (prop.l2CacheSize / 1024) << " kB" << endl;
			cout << "Global Caching in L1 = " << prop.globalL1CacheSupported << endl;
			cout << "Local Caching in L1 = " << prop.localL1CacheSupported << endl;
		}

		switch (prop.computeMode)
		{
		case 0:
			cout << "Default Compute Mode" << endl; // Device is not restricted and multiple threads can use cudaSetDevice() with this device
			break;
		case 1:
			cout << "Exclusive Compute Mode" << endl; // Only one thread will be able to use cudaSetDevice() with this device
			break;
		case 2:
			cout << "Prohibited Compute Mode" << endl;
			break;
		}

		cout << "Global Memory = " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
		cout << "Shared Memory = " << prop.sharedMemPerBlock / 1024 << " kB" << endl;
		cout << "Shared Memory per SM = " << prop.sharedMemPerMultiprocessor / 1024 << "kB" << endl;
		cout << "Constant Memory = " << prop.totalConstMem / 1024 << " kB" << endl;

		cout << "Maximum Blocks/SM = " << prop.maxBlocksPerMultiProcessor << endl;

		cout << "Maximum # of Threads/SM = " << prop.maxThreadsPerMultiProcessor << endl;
		cout << "Maximum # of Threads/Block = " << prop.maxThreadsPerBlock << endl;
		cout << "Maximum # of Threads along each Dimension of the Block" << endl;
		cout << '\t' << "X = " << prop.maxThreadsDim[0] << '\t' << "Y = " << prop.maxThreadsDim[1] << '\t' << "Z = " << prop.maxThreadsDim[2] << endl;

		cout << "Maximum # of blocks allowed along each Dimension of the Grid" << endl;
		cout << '\t' << "X = " << prop.maxGridSize[0] << '\t' << "Y = " << prop.maxGridSize[1] << '\t' << "Z = " << prop.maxGridSize[2] << endl;
		cout << "Registers Per Block = " << prop.regsPerBlock << endl;
		cout << "Registers per SM = " << prop.regsPerMultiprocessor << endl;
		cout << "Warp Size  = " << prop.warpSize << " Threads" << endl;


		cout << "Can execute Concurrent Kernels = " << prop.concurrentKernels << endl;
		cout << "Can perform overlap = " << prop.deviceOverlap << endl;
		cout << "Host Memory Mapping = " << prop.canMapHostMemory << endl;
		cout << "Runtime limit on Kernels = " << prop.kernelExecTimeoutEnabled << endl;

		cout << "Single Precision to Double Precision Performance Ratio = " << prop.singleToDoublePrecisionPerfRatio << endl;
		cout << "Memory Allocation Alignment = " << prop.textureAlignment << endl;

		cudaSharedMemConfig config;
		cudaDeviceGetSharedMemConfig(&config);
		if (config == cudaSharedMemBankSizeDefault)
		{
			cout << "Default" << endl;
		}
		if (config == cudaSharedMemBankSizeEightByte)
		{
			cout << "Default four Byte" << endl;
		}
		cout << config << endl;
	}
	int driverVersion;
	cudaDriverGetVersion(&driverVersion);
	cout << "Driver Version: " << driverVersion << endl;
	return 0;
}