#include "WarpFunc.h"

int main()
{
	int* h_A{};

	//Memory Allocation
	h_A = new int[SIZE];

	//Initialize Data
	InitializeVector(h_A, SIZE);

	chrono::time_point<std::chrono::system_clock> start, end;
	//CPU Odd Count Computation
	int odd_count = 0;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < SIZE; i++)
	{
		if (h_A[i] % 2) //Check if its odd using modulo operation
		{
			odd_count++;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Sequential Execution time: " << (elasped_seconds.count() * 1000000.0f) << " usecs" << endl;
	cout << "Number of Odds determined by the CPU: " << odd_count << endl;

	//Call the GPU helper function using the shared memory kernel
	CountOddsHelper(h_A, odd_count, SIZE);

	//Call the GPU helper function using the shared memory with Warp Primitives kernel
	CountOddsWPHelper(h_A, odd_count, SIZE);
	
	//Deallocate Memory
	delete[] h_A;

	return 0;
}