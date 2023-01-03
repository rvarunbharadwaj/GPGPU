#include "ParallelReduction.h"

int main()
{

	float* vector = new float[VECTOR_SIZE];
	float* vectorCopy = new float[VECTOR_SIZE];

	OnInitializeInputData(vector,VECTOR_SIZE);
	CopyInputData(vectorCopy, vector, VECTOR_SIZE);

	float sum = OnSequentialReduce(vector, VECTOR_SIZE);
	cout << "\t\tSequential Reduction: " << sum << endl;

	sum = 0.0f;
	chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	sum = OnCPURecursiveReduce(vector, VECTOR_SIZE);
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elasped_seconds = end - start;
	cout << "CPU Recursive Reduction Execution time: " << (elasped_seconds.count() * 1000.0f) << " msecs" << endl;
	cout << "\t\tRecursive Reduction: " << sum << endl;

	//GPU Executions
	//Neighborhood reduction with divergence
	OnNeighborhood(vectorCopy);

	delete[] vector;
	delete[] vectorCopy;
	return 0;
}