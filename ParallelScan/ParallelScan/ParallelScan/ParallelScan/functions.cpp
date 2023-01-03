#include "ParallelScan.h"

void OnInitializeInputData(float* vectorTemp, int SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		vectorTemp[i] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}

void CopyInputData(float* vectorTemp, float* ref,int SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		vectorTemp[i] = ref[i];
	}
}

void ZeroData(float* in, int SIZE)
{
	for (int i = 0; i < SIZE; i++)
	{
		in[i] = 0.0f;
	}
}

void PrintVectors(float* vector, int SIZE)
{
	for (int i = 0; i < 4; i++)
	{
		cout << vector[i] << '\t';
	}
	cout << ", . . .\t";
	for (int i = SIZE - 1; i > SIZE - 4; i--)
	{
		cout << vector[i] << '\t';
	}
	cout << endl;
}

void Verify(float* ref, float* in, int SIZE)
{
	float fTolerance = 1.0E-04f;
	for (int i = 0; i < SIZE; i++)
	{
		if (fabs(ref[i] - in[i]) > fTolerance)
		{
			cout << "Error" << endl;
			cout << "\vectRef[" << (i + 1) << "] = " << ref[i] << endl;
			cout << "\vectGPU[" << (i + 1) << "] = " << in[i] << endl;
			return;
		}
	}
}