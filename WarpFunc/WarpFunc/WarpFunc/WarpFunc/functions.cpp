#include "WarpFunc.h"

void InitializeVector(int* vect, const int nSize)
{
	const int range_from = 1;
	const int range_to = 10000;
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_int_distribution<int>  distr(range_from, range_to);
	for (int i = 0; i < nSize; i++)
	{
		//vect[i] = distr(generator);
		vect[i] = 1;
	}
}

