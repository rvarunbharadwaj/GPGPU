#include "ParallelReduction.h"

float OnSequentialReduce(float* vectorTemp, const int SIZE)
{
	float sum = 0.0f;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < SIZE; i++)
	{
		sum += vectorTemp[i];
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Sequential Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	return sum;
}

//Recursive function
float OnCPURecursiveReduce(float* VectorTemp, const int SIZE)
{
	if (SIZE == 1)
	{
		return VectorTemp[0];
	}

	int const stride = SIZE / 2;

	for (int i = 0; i < stride; i++)
	{
		VectorTemp[i] += VectorTemp[i + stride];
	}
	return OnCPURecursiveReduce(VectorTemp, stride);
}
