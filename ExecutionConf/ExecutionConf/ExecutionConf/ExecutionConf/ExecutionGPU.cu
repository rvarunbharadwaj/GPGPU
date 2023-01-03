#include "ExecutionConf.h"

void calcExecConf(int numberOfThreads, 
	int registersPerThread, 
	int sharedPerThread, 
	int& bestThreadsPerBlock, 
	int& bestTotalBlocks)
{
	cudaDeviceProp pr; //get device properties
	cudaGetDeviceProperties(&pr, 0);     // replace 0 with appropriate ID in case of a multi-GPU system

	int maxRegs = pr.regsPerBlock;
	int SM = pr.multiProcessorCount;
	int warp = pr.warpSize;
	int sharedMem = pr.sharedMemPerBlock;
	int maxThreadsPerSM = pr.maxThreadsPerMultiProcessor;
	int totalBlocks;
	float imbalance, bestimbalance;
	int threadsPerBlock;
	int numWarpSchedulers = 4;

	bestimbalance = SM;

	// initially calculate the maximum possible threads per block. Incorporate limits imposed by :
  // 1) SM hardware 
	threadsPerBlock = maxThreadsPerSM;
	// 2) Use the registers size limits
	threadsPerBlock = min(threadsPerBlock, maxRegs / registersPerThread);
	// 3) use the shared memory size limits
	if (sharedPerThread > 0)
	{
		threadsPerBlock = min(threadsPerBlock, sharedMem / sharedPerThread); //Using too much shared mem, we need to decrease the number of threads
	}

	// make the number of threads/block a multiple of warpSize  
	int tmp = threadsPerBlock / warp;
	threadsPerBlock = (tmp + 1) * warp;//Previous line is int division, hence we add 1 and multiply by warpsize to make it a multiple
	cout << "threadPerBlock" << '\t' << "totalBlocks" << '\t' << "imbalance" << '\t' << "bestimbalance" << endl;

	//Loop until the threadsPerBlock is greater than the product of Number of WarpSchedulers and imbalance is not equal to zero
	for (; threadsPerBlock >= numWarpSchedulers * warp && bestimbalance != 0; threadsPerBlock -= warp)
	{
		totalBlocks = (int)ceil(1.0 * numberOfThreads / threadsPerBlock); //Mutiply by 1.0 to make it a double
		if (totalBlocks % SM == 0) //It's a mutiple of SM, i.e. we have distributed work evenly
		{
			imbalance = 0;
		}

		else
		{
			int blocksPerSM = totalBlocks / SM; //SUm of the SM's will get this number of blocks and others will get +1 block
			imbalance = (SM - (totalBlocks % SM)) / (blocksPerSM + 1.0);
		}
		cout << threadsPerBlock << "\t\t" << totalBlocks << "\t\t" << imbalance << "\t\t" << bestimbalance << endl;
		if (bestimbalance >= imbalance)
		{
			bestimbalance = imbalance;
			bestThreadsPerBlock = threadsPerBlock;
			bestTotalBlocks = totalBlocks;
		}
	}
}