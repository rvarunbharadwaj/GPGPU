#include "TransposeSMEM.h"

int main()
{
	srand((unsigned)time(NULL));

	//Setup Matrix
	int cols = 1 << 12;
	int rows = 1 << 12;
	cout << "Matrix Size = " << rows << " x " << cols << endl;

	float* Matrix = new float[rows * cols];
	float* CPUMatrixTranspose = new float[rows * cols];
	float* GPUMatrixTranspose = new float[rows * cols];

	InitializeMatrix(Matrix, rows, cols);
	ZeroMatrix(CPUMatrixTranspose, rows, cols);
	TransposeOnCPU(Matrix, CPUMatrixTranspose, rows, cols);

	ZeroMatrix(GPUMatrixTranspose, rows, cols);
	TransposeOnGPU(Matrix, GPUMatrixTranspose, CPUMatrixTranspose, rows, cols);

	delete[] Matrix;
	delete[] CPUMatrixTranspose;
	delete[] GPUMatrixTranspose;
	return 0;
}