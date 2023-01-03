#include "VectorAddition.h"

void CPUVectorAddition(float* vectA, float* vectB, float* vectC, const int nSize)
{
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < nSize; i++)
	{
		vectC[i] = vectA[i] + vectB[i];
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Sequential Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
}

