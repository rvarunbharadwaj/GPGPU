#include "RGBtoGrayScale.h"

float cpu__RGBtoGrayScale_Ver0(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg)
{
	int height = rgbImg.height();
	int width = rgbImg.width();
	
	//Naive Implementation
	auto start = high_resolution_clock::now();
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			grayImg(x, y) = 0.21 * rgbImg(x, y, 0, 0) + 0.71 * rgbImg(x, y, 0, 1) + 0.07 * rgbImg(x, y, 0, 2); //third value is used when we stack frames in 3d, 4th parameter is the channels
		}
	}
	auto end = high_resolution_clock::now();
	auto elapsed_seconds = end - start;
	return duration_cast<microseconds>(elapsed_seconds).count();
}

float cpu__RGBtoGrayScale_Ver1(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w)
{
	unsigned int stride = h * w;
	unsigned int stride2 = 2 * stride;
	unsigned char* r = in;
	unsigned char* g = in + stride;
	unsigned char* b = in + stride2;

	auto start = high_resolution_clock::now();
	for (int i = 0; i < h*w; i++)
	{
		*out = 0.21f * (*r) + 0.71f * (*g) + 0.07f * (*b);
		out++;
		r++;
		g++;
		b++;
	}
	auto end = high_resolution_clock::now();
	auto elasped_seconds = end - start;

	return duration_cast<microseconds>(elasped_seconds).count();
}