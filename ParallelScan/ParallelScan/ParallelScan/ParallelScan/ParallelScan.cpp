#include "ParallelScan.h"

int main()
{
	float* input = new float[VECTOR_SIZE];
	float* outRef = new float[VECTOR_SIZE];
	float* scan_output = new float[VECTOR_SIZE];

	OnInitializeInputData(input,VECTOR_SIZE);
	//CopyInputData(vectorCopy, vector, VECTOR_SIZE);
	cout << "Input Vector" << endl;
	PrintVectors(input, VECTOR_SIZE);
	OnSequentialScan(input,scan_output, VECTOR_SIZE);
	cout << "Scanned Output" << endl;
	PrintVectors(scan_output, VECTOR_SIZE);
	CopyInputData(outRef, scan_output, VECTOR_SIZE);

	ZeroData(scan_output, VECTOR_SIZE);
	//GPU Inefficient Execution
	Helper_Scan(input, scan_output, outRef, VECTOR_SIZE);
	
	delete[] input;
	delete[] outRef;
	delete[] scan_output;
	return 0;
}