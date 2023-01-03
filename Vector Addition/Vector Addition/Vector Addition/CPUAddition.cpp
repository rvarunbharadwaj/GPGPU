#include "VectorAddition.h"

void CPUVectorAddition(float* vectA, float* vectB, float* vectC, const int nSize)
{
	for (int i = 0; i < nSize; i++)
	{
		vectC[i] = vectA[i] + vectB[i];
	}
}

