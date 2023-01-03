#include "TransposeBM.h"

void TransposeOnCPU(float* matrix, float* matrixTranspose, int ny, int nx)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int y = 0; y < ny; y++)
	{
		for (int x = 0; x < nx; x++)
		{
			//Load Coalesced and Store stride
			matrixTranspose[x * ny + y] = matrix[y * nx + x]; //out of place transpose
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Transpose Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}