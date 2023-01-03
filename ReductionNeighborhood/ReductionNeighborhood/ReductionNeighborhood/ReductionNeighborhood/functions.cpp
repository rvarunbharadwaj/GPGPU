#include "ParallelReduction.h"

void OnInitializeInputData(float* vectorTemp, const int SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		vectorTemp[i] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}

void CopyInputData(float* vectorTemp, float* ref,const int SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		vectorTemp[i] = ref[i];
	}
}