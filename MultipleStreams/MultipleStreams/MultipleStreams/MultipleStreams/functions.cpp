#include "VectorAddition.h"

void InitializeVector(float* vect, const int nSize)
{
	const float range_from = 1.00f;
	const float range_to = -1.00f;
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_real_distribution<float>  distr(range_from, range_to);
	for (int i = 0; i < nSize; i++)
	{
		vect[i] = distr(generator);
	}
}

void DisplayVector(string name, float* vect, const int nSize)
{
	cout << name << endl;
	if (nSize > 10)
	{
		for (int i = 0; i < 5; i++)
		{
			cout << vect[i] <<',';
		}
		for (int i = nSize - 5; i < nSize; i++)
		{
			cout << vect[i] <<',';
		}
		cout << endl;
	}
	else
	{
		cout << name << ": ";
		for (int i = 0; i < nSize; i++)
		{
			cout << vect[i] << ',';
		}
		cout << endl;
	}
}

void VerifyResults(float* vectRef, float* vectGPU, const int nSize)
{
	float fTolerance = 1.0E-04f;
	for (int i = 0; i < nSize; i++)
	{
		if (fabs(vectRef[i] - vectGPU[i]) > fTolerance)
		{
			cout << "Error" << endl;
			cout << "vectRef[" << (i + 1) << "] = " << vectRef[i] << endl;
			cout << "vectGPU[" << (i + 1) << "] = " << vectGPU[i] << endl;
			return;
		}
	}
}