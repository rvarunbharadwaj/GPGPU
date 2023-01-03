#include "Prob4.h"
#include "GPUErrors.h"

int main()
{
	srand((unsigned)time(NULL));

	/*Matrix Size(n x n)
	Run the program for sizes 256, 512, 1024, 2048, and 4096*/
	int n = 1 << 12;
	
	cout << "Multiplication of Matrix of size: " << n << "x" << n <<"with "<<" Vector of size: "<<n<<"x"<<1<< endl;
	float* M, * V, * P;

	M = new float[n * n];
	V = new float[n];
	P = new float[n];

	InitializeData(M,V,n);

	//Host Matrix Vector Multiplication
	cpuMatrixVectorMult(M, V, P, n);


	float* gpuP;
	gpuP = new float[n];
	gpuMultHelper(M, V, gpuP, n);
	Verification(P, gpuP, n);
	
	//Save the value of matrix, vector and the resultant vector to a csv file
	WriteData("CPU.csv", M, V, P, n);

	delete [] M;
	delete[] V;
	delete[] P;
	delete[] gpuP;

	return 0;
}