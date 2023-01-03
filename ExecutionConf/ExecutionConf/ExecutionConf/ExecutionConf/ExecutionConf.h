#pragma once
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;
//to optimze the number of thereads and blocks
void calcExecConf(int numberOfThreads, 
	int registersPerThread, 
	int sharedPerThread, 
	int& bestThreadsPerBlock, 
	int& bestTotalBlocks);
