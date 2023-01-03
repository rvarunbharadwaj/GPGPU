#include <iostream>
#include <cmath>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

int main() 
{
	//Declare a CUDA Device property structure variable
	cudaDeviceProp prop;
	//An integer variable to store the number of GPUs
	int nCountDevices{};

	//Query the RT system about the number of GPUs
	cudaGetDeviceCount(&nCountDevices);
	cout << "Number of GPUs " << nCountDevices << endl;

	for (int i = 0; i < nCountDevices; i++)
	{
		//Call the get device api and store the information in the structure variable: prop
		cudaGetDeviceProperties(&prop, i);
		cout << "Name = " << prop.name << endl;
		//Compute Capability of GPU
		cout << "Compute Capability " << prop.major << "." << prop.minor << endl;
		cout << "Streaming Multiprocessors = " << prop.multiProcessorCount << endl;
		cout << "Streaming Processors/SM(SP) = " << _ConvertSMVer2Cores(prop.major, prop.minor) << endl;
		cout << "GPU CLock Rate: " << (prop.clockRate / pow(10.0, 3.0)) << " MHz" << endl;
		cout << "Global Memory CLock Rate: " << (prop.memoryClockRate / pow(10.0, 3.0)) << " MHz" << endl;
		cout << "Memory Bus Width = " << prop.memoryBusWidth << " bits" << endl;
		cout << "Global Memory = " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
		cout << "Maximum # Blocks/SM = " << prop.maxBlocksPerMultiProcessor << endl;
		cout << "Maximum Threads/SM = " << prop.maxThreadsPerMultiProcessor << endl;
		cout << "Maximum Threads/Block = " << prop.maxThreadsPerBlock << endl;
	}
	return 0;
}