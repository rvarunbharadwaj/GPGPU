#include "MatrixMult.h"

void InitializeMatrix(float *matrix, int ny, int nx)
{
	float *p = matrix;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = ((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		p += nx;
	}
}

void ZeroMatrix(float *temp, const int ny, const int nx)
{
	float *p = temp;

	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			p[j] = 0.0f;
		}
		p += nx;
	}
}


void DisplayMatrix(string name, float* temp, const int ny, const int nx)
{
	if (ny < 6 && nx < 6)
	{
		cout << name << endl;
		for (int i = 0; i < ny; i++)
		{
			for (int j = 0; j < nx; j++)
			{
				cout << setprecision(6) << temp[(i * nx) + j] << "\t";
			}
			cout << endl;
		}
	}
}

void MatrixMultVerification(float* hostC, float* gpuC, const int ny, const int nx)
{
	float fTolerance = 1.0E-05;
	float* p = hostC;
	float* q = gpuC;
	for (int i = 0; i < ny; i++)
	{
		for (int j = 0; j < nx; j++)
		{
			if (fabs(p[j] - q[j]) > fTolerance)
			{
				cout << "Error" << endl;
				cout << "\thostC[" << (i + 1) << "][" << (j + 1) << "] = " << p[j] << endl;
				cout << "\tgpuC[" << (i + 1) << "][" << (j + 1) << "] = " << q[j] << endl;
				return;
			}
		}
		p += nx;
		q += nx;
	}
}