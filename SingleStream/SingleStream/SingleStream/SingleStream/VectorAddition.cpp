#include "VectorAddition.h"

int main()
{
	float* h_A, * h_B, * h_C_CPU, * h_C_GPU;

	//Memory Allocation
	h_A = new float[SIZE];
	h_B = new float[SIZE];
	h_C_CPU = new float[SIZE];
	h_C_GPU = new float[SIZE];

	//Initialize Data
	InitializeVector(h_A, SIZE);
	InitializeVector(h_B, SIZE);

	CPUVectorAddition(h_A, h_B, h_C_CPU, SIZE);

	//Perform Vector Addition on GPU with Default Stream
	cout << "GPU Memory Used: " << (3 * VECTOR_SIZE_IN_BYTES / (pow(1024, 3))) << "GB" << endl;
	WithDefaultStream(h_A, h_B, h_C_GPU, SIZE);
	VerifyResults(h_C_CPU, h_C_GPU, SIZE);

	//Deallocate Memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C_CPU;
	delete[] h_C_GPU;

	//Perform Vector Addition on GPU with Explicit Stream
	WithExplicitStream(SIZE);
	

	return 0;
}