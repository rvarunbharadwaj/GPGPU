#include "ParallelScan.h"

void OnSequentialScan(float* in, float* out, int SIZE)
{
	float sum = 0.0f;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	out[0] = in[0];
	for (int i = 1; i < SIZE; i++)
	{
		out[i] = out[i - 1] + in[i];
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Sequential Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}

