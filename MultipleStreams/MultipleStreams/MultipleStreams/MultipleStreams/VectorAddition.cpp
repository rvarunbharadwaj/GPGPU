#include "VectorAddition.h"

int main()
{
	float* h_A{}, * h_B{}, * h_C_CPU{}, * h_C_GPU{};

	//Memory Allocation
	h_A = new float[SIZE];
	h_B = new float[SIZE];
	h_C_CPU = new float[SIZE];

	//Initialize Data
	InitializeVector(h_A, SIZE);
	InitializeVector(h_B, SIZE);

	CPUVectorAddition("Pageable",h_A, h_B, h_C_CPU, SIZE);

	//Deallocate Memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C_CPU;

	//Perform Vector Addition on GPU with Multiple Streams without Scheduling 
	cout << "GPU Memory Used: " << (3 * VECTOR_SIZE_IN_BYTES / (pow(1024, 3))) << "GB" << endl<<endl;
	cout << "Executing Multiple Streams without Scheduling" << endl;
	MultipleStreamsWOScheduling(SIZE);

	//Perform Vector Addition on GPU with Multiple Streams and Scheduling
	cout <<endl<< "Executing Multiple Streams with Scheduling" << endl;
	MultipleStreamsWithScheduling(SIZE);

	return 0;
}