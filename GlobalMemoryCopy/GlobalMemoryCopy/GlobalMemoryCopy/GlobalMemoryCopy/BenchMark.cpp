#include "PerformanceBounds.h"

int main()
{
	srand((unsigned)time(NULL));

	//Setup Matrix
	int cols = 1 << 12;
	int rows = 1 << 12;
	cout << "Matrix Size = " << rows << " x " << cols << endl;
	float* Matrix = new float[rows * cols];

	InitializeMatrix(Matrix, rows, cols);
	PerformanceBounds(Matrix, rows, cols);

	delete[] Matrix;
	return 0;
}
