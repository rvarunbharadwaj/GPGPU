#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
using namespace std;
using namespace std::chrono; //Higher resoultion clock
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Image Processing Routines
#include "CImg.h"
using namespace cimg_library;

//GPU Error Handling
#include "GPUErrors.h"

//CPU Functions
float cpu__RGBtoGrayScale_Ver0(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg); //we use & to referencing the image
float cpu__RGBtoGrayScale_Ver1(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w); //passing pointers

//GPU Helper Function
__host__ double gpu_RGBtoGrayScaleHelper(unsigned char* h_in, unsigned char* h_out, unsigned int rgbSIZE,
	unsigned int graySIZE,
	unsigned int h, 
	unsigned int w,
	unsigned int kernelVer);

//GPU Kernels
__global__ void gpu_RGBtoGrayScaleVer0(unsigned char* in, unsigned char* out, unsigned int h, unsigned int w);
__global__ void gpu_RGBtoGrayScaleVer1(unsigned char* r, 
	unsigned char* g,
	unsigned char* b,
	unsigned char* out, unsigned int h, unsigned int w);
__global__ void gpu_RGBtoGrayScaleVer2(unsigned char* r,
	unsigned char* g,
	unsigned char* b,
	unsigned char* out, unsigned int h, unsigned int w);





