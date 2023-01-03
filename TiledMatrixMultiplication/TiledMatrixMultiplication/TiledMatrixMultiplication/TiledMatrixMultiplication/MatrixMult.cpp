#include "MatrixMult.h"
#include "GPUErrors.h"

int main()
{
	srand((unsigned)time(NULL));

	int rows = 1 << 10;
	int cols = 1 << 10;
	cout << "Matrix Multiplication of Size: " << rows << "x" << cols << endl;
	float* A, * B, * C;

	A = new float[rows * cols];
	B = new float[rows * cols];
	C = new float[rows * cols];

	InitializeMatrix(A, rows, cols);
	InitializeMatrix(B, rows, cols);

	//Host Multiplication
	cpuMatrixMult(A, B, C, rows, cols);

	DisplayMatrix("A", A, rows, cols);
	DisplayMatrix("B", B, rows, cols);
	DisplayMatrix("C", C, rows, cols);

	float* gpuC;
	gpuC = new float[rows * cols];
	//Array to store the matrix product from tiled multiplication
	float* gpuC_Tiled = new float[rows * cols];

	gpuMultHelper(A, B, gpuC, gpuC_Tiled,C, rows, cols);

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] gpuC;
	delete[] gpuC_Tiled;

	return 0;
}