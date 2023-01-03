#include "Prob4.h"

void cpuMatrixVectorMult(float* matrix, float* v, float* p, const int Size)
{
	float fSum; //Temp value to store the summation
	float* m = matrix; //Poniter to the matrix
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};
	start = high_resolution_clock::now();
	//Write code to perform Matrix Vector Multiplication
	//First implementation of Matrix Vector Multiplication
	/*for (int i = 0; i < Size; i++)
	{
		fSum = 0.0f;
		for (int j = 0; j < Size; j++)
		{
			fSum += v[j] * matrix[i * Size + j];
		}
		p[i] = fSum;
	}*/

	//Optimised version of CPU Matrix Vector Multiplication
	for (int i = 0; i < Size; i++)
	{
		float* vt = v; //Pointer to the Vector
		fSum = 0.0f; //Initialize to zero every iteration
		for (int j = 0; j < Size; j++)
		{
			fSum += (*vt++) * (*m++); //vt and m point to v and matrix, after each iteration the address is incremented to move to the next value
		}
		p[i] = fSum; //Storing the value in output Vector
	}

	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "CPU Execution time: " << computeTime << " usecs" << endl;
}