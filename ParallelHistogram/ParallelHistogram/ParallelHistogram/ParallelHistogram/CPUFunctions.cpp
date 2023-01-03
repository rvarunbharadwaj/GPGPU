#include "ParallelHistogram.h"

void cpu__RGBtoGrayScale(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg)
{
	int height = rgbImg.height();
	int width = rgbImg.width();
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			grayImg(x, y) = 0.21f * rgbImg(x, y, 0, 0) + 0.71f * rgbImg(x, y, 0, 1) + 0.07f * rgbImg(x, y, 0, 2);
		}
	}
}

//A function construct the histogram of a grayscale image
void cpu_Histogram(unsigned char* in, unsigned int bins, unsigned int* hist, unsigned int h, unsigned int w)
{
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};
	unsigned int N = h * w; //pixels in an image
	//Initialize the histogram counts
	for (unsigned int i = 0; i < bins; i++)
	{
		hist[i] = 0;
	}
	//Compute the histogram
	start = high_resolution_clock::now();
	for (unsigned int i = 0; i < N; i++)
	{
		hist[in[i]]++; 
	}
	end = high_resolution_clock::now();
	auto elapsed_seconds = end - start;
	computeTime = duration_cast<microseconds>(elapsed_seconds).count();
	cout << "Sequential Histogram: CPU Execution time: " << computeTime << " usecs" << endl;
}

//A function verify Histogram Computation
void Verify(unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins)
{
	for (unsigned int i = 0; i < bins; i++)
	{
		if (cpu_histogram[i] != gpu_histogram[i])
		{
			cout << "Error in bin: " << i << " CPU Bin Value:" << cpu_histogram[i] << " GPU Bin Value:" << gpu_histogram[i] << endl;
			return;
		}
	}
}

//A Function to write the histogram data to a file
void WriteHistograms(string FileName, unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins)
{
	fstream outfile;
	outfile.open(FileName, std::ios_base::out);
	for (int i = 0; i < bins; i++)
	{
		outfile << i << "," << cpu_histogram[i] << "," << gpu_histogram[i] << endl;
	}
	outfile.close();
}

