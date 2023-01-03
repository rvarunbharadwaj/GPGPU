#include "VectorAddition.h"

void InitializeVector(float* vect, const int nSize) //Random number between +1 and -1
{
	for (int i = 0; i < nSize; i++)
	{
		vect[i] = ((float)rand() / (float)((RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN));
	}
}

void DisplayVector(string name, float* vect, const int nSize) //For comparing results
{
	if (nSize > 10)
	{
		for (int i = 0; i < 5; i++)
		{
			cout << vect[i] << '\t';
		}
		for (int i = nSize - 5; i < nSize; i++)
		{
			cout << vect[i] << '\t';
		}
		cout << endl;
	}
	else
	{
		cout << name << ": ";
		for (int i = 0; i < nSize; i++)
		{
			cout << vect[i] << '\t';
		}
		cout << endl;
	}
}

void VerifyResults(float* vectRef, float* vectGPU, const int nSize)
{
	float fTolerance = 1.0E-04f; //upto 4 decimal places
	for (int i = 0; i < nSize; i++)
	{
		if (fabs(vectRef[i] - vectGPU[i]) > fTolerance)
		{
			cout << "Error" << endl;
			cout << "\vectRef[" << (i + 1) << "] = " << vectRef[i] << endl;
			cout << "\vectGPU[" << (i + 1) << "] = " << vectGPU[i] << endl;
			return;
		}
	}
}