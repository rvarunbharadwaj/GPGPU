#include "TransposeSMEM.h"

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

void ZeroMatrix(float* temp, const int ny, const int nx)
{
	float* p = temp;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = 0.0f;
		}
		p += nx;
	}
}

void VerifyTranspose(float* gpuTranspose, float* cpuTranspose, int ny, int nx)
{
	float* p = gpuTranspose;
	float* pTranspose = cpuTranspose;
	float fTol = 1E-06;
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			if (fabsf(gpuTranspose[j] - cpuTranspose[j]) > fTol)
			{
				cout << "Transpose Error" << endl;
				cout << '\t' << "CPU Element[" << (i + 1) << "][" << (j + 1) << "] = " << cpuTranspose[j] << endl;
				cout << '\t' << "GPU Element[" << (i + 1) << "][" << (j + 1) << "] = " << gpuTranspose[j] << endl;
				return;
			}
		}
	}
}