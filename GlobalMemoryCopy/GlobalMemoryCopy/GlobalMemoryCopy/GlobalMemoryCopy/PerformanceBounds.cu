#include "PerformanceBounds.h"
#include "GPUErrors.h"

__global__ void CopyRowWise(float* g_Matrix, float* g_MatrixCopy, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)
	{
		g_MatrixCopy[iy * nx + ix] = g_Matrix[iy * nx + ix]; //reading across the row and writing across the row
	}
}

__global__ void CopyColWise(float* g_Matrix, float* g_MatrixCopy, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)
	{
		g_MatrixCopy[ix * ny + iy] = g_Matrix[ix * ny + iy]; //reading across the col and write across the col
	}

}

__host__ void PerformanceBounds(float *h_Matrix, int ny, int nx)
{
	float *d_Matrix;
	float *d_MatrixCopy;

	float *h_MatrixCopy = new float[ny*nx];
	const int MatrixSizeInBytes = ny * nx * sizeof(float);

	//Allocate memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_Matrix, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_MatrixCopy, MatrixSizeInBytes));

	//transfer data from CPU Memory to GPU Memory
	HandleCUDAError(cudaMemcpy(d_Matrix, h_Matrix, MatrixSizeInBytes, cudaMemcpyHostToDevice));

	//Block and Grid Parameters
	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
	cout << "2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	//Matrix Copy Row wise - Coalesced Access
	CopyRowWise << <grid, block >> > (d_Matrix, d_MatrixCopy, ny, nx);
	cudaDeviceSynchronize();
	HandleCUDAError(cudaMemcpy(h_MatrixCopy, d_MatrixCopy, MatrixSizeInBytes, cudaMemcpyDeviceToHost));

	//Matrix Copy Column wise - Stride Access
	CopyColWise << <grid, block >> > (d_Matrix, d_MatrixCopy, ny, nx);
	cudaDeviceSynchronize();
	HandleCUDAError(cudaMemcpy(h_MatrixCopy, d_MatrixCopy, MatrixSizeInBytes, cudaMemcpyDeviceToHost));

	delete[] h_MatrixCopy;
	HandleCUDAError(cudaFree(d_Matrix));
	HandleCUDAError(cudaFree(d_MatrixCopy));
	HandleCUDAError(cudaDeviceReset());
}

