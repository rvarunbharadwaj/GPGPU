#include "PerformanceBounds.h"

void InitializeMatrix(float* matrix, int ny, int nx)
{
	float* p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		p += nx;
	}
}