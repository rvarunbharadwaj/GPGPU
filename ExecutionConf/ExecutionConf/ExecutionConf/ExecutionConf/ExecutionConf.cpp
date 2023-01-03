#include "ExecutionConf.h"

int main()
{
	int registersPerThread{};
	int sharedPerThread{};
	int numberOfThreads{};
	int bestThreadsPerBlock{}, bestTotalBlocks{};

	cout << "Enter the desired number of threads: ";
	cin >> numberOfThreads;

	cout << "Enter registers per thread: ";
	cin >> registersPerThread;

	cout << "Enter sharedPerThread: ";
	cin >> sharedPerThread;

	//Call the calcExecConf function
	calcExecConf(numberOfThreads, registersPerThread, sharedPerThread, bestThreadsPerBlock, bestTotalBlocks);
	cout<<"BEST grid with "<<bestTotalBlocks<<" blocks and each with "<<bestThreadsPerBlock<<" threads"<<endl;

	return 0;
}