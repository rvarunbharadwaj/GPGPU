#include "Prob4.h"

void InitializeData(float* matrix, float* v, const int Size)
{
	float *p = matrix;

	for (int i = 0; i < Size; i++)
	{
		for (int j = 0; j < Size; j++)
		{
			p[j] = ((float)rand() / (RAND_MAX + 1)*(RANGE_MAX - RANGE_MIN) + RANGE_MIN);
		}
		p += Size;
		v[i] = ((float)rand() / (RAND_MAX + 1) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN);
	}
}

void Verification(float* vect, float* gpuVect, const int Size)
{
	float fTolerance = 1.0E-05;
	float* p = vect;
	float* q = gpuVect;
	for (int j = 0; j < Size; j++)
	{
		if (fabs(p[j] - q[j]) > fTolerance)
		{
			cout << "Error" << endl;
			cout << "\thostVect[" << (j + 1) << "] = " << p[j] << endl;
			cout << "\gpuVect[" << (j + 1) << "] = " << q[j] << endl;
			return;
		}
	}	
}

void WriteData(string FileName, float *matrix, float* v, float *p,const int nSize)
{
	fstream outfile;
	float* M = matrix;
	if (nSize <= 4)
	{
		outfile.open(FileName, std::ios_base::out);
		outfile << "M" << endl;
		for (int i = 0; i < nSize; i++)
		{
			for (int j = 0; j < nSize; j++)
			{
				outfile << M[j]<<",";
			}
			outfile << endl;
			M += nSize;
		}
		outfile << "V" << endl;
		for (int i = 0; i < nSize; i++)
		{
			outfile << v[i] << endl;
		}
		outfile << "P" << endl;
		for (int i = 0; i < nSize; i++)
		{
			outfile << p[i] << endl;
		}
		outfile.close();
	}
}