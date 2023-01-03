#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//Image Processing Routines
#include "CImg.h"
using namespace cimg_library;

//GPU Error Handling
#include "GPUErrors.h"

//CPU Functions
void cpu__RGBtoGrayScale(CImg<unsigned char>& rgbImg, CImg<unsigned char>& grayImg);
void cpu_Histogram(unsigned char* in, unsigned int bins, unsigned int* hist, unsigned int h, unsigned int w);
void Verify(unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins);
void WriteHistograms(string FileName, unsigned int* cpu_histogram, unsigned int* gpu_histogram, unsigned int bins);

//GPU Helper Function
__host__ void gpu_HistogramHelper(unsigned char* h_in, 
	unsigned int* histogram,
	unsigned int graySIZE,
	unsigned int h, 
	unsigned int w,
	unsigned int BinSize,
	unsigned int* cpu_hist);

//GPU Kernels
__global__ void gpu_Histogram(unsigned char* in, unsigned int* out, unsigned int h, unsigned int w, unsigned int SIZE);
__global__ void gpu_HistogramVer2(unsigned char* in, unsigned int* out, unsigned int h, unsigned int w, unsigned int SIZE);
__global__ void gpu_SharedWithThreadCoarse(unsigned char* in, int* out, unsigned int h, unsigned int w, unsigned int SIZE, unsigned int CFACTOR); //CFATOR - coarseining factor
__global__ void gpu_SharedWithThreadCoarseAggregation(unsigned char* in, int* out, unsigned int h, unsigned int w, unsigned int SIZE, unsigned int CFACTOR); //CFATOR - coarseining factor




