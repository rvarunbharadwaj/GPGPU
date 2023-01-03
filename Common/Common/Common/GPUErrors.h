#pragma once
#include <iostream>
#include <iomanip>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//GPU Error Functions
bool HandleCUDAError(cudaError_t t);
bool GetCUDARunTimeError();
