#include "MatrixMult.h"

void cpuMatrixMult(float* A, float* B, float* C, const int ny, const int nx)
{
	float fSum;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			fSum = 0.0f;
			for (int k = 0; k < nx; k++)
			{
				fSum += (A[(i * nx) + k] * B[(k * nx) + j]);
			}
			C[(i * nx) + j] = fSum;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}