#include "VectorAddition.h"

int main()
{
	float* h_A, * h_B, * h_C_CPU, * h_C_GPU;
	chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double>elapsed_seconds;

	//Memory Allocation
	h_A = new float[SIZE];
	h_B = new float[SIZE];
	h_C_CPU = new float[SIZE];
	h_C_GPU = new float[SIZE];

	//Initialize Data
	InitializeVector(h_A, SIZE);
	InitializeVector(h_B, SIZE);

	//Perform Vector Addition on CPU (host)
	start = std::chrono::system_clock::now();
	CPUVectorAddition(h_A, h_B, h_C_CPU, SIZE);
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start; //measures system ticks
	cout << "CPU execution time = " << (elapsed_seconds.count() * 1000.0f) << "msecs" << endl; //.count() will return the seconds

	DisplayVector("A", h_A, SIZE);
	DisplayVector("B", h_B, SIZE);
	DisplayVector("C", h_C_CPU, SIZE);
	
	cout << "GPU memory used: " << (3 * VECTOR_SIZE_IN_BYTES / pow(1024, 3)) <<"GB"<< endl;
	//Perform Vector Addition on GPU
	GPUAdditionHelper(h_A, h_B, h_C_GPU, SIZE);
	//Verify Results
	VerifyResults(h_C_CPU, h_C_GPU, SIZE);
	delete[] h_A;
	delete[] h_B;
	delete[] h_C_CPU;
	delete[] h_C_GPU;

	return 0;
}