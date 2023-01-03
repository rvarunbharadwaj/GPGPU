#include "TransposeBM.h"
#include "GPUErrors.h"

__global__ void NaiveRowTranspose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)
	{
		g_MatrixTranspose[ix * ny + iy] = g_Matrix[iy * nx + ix]; //read coalsced and write stride (out of place)
	}
}

__global__ void NaiveColTranspose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)
	{
		g_MatrixTranspose[iy * nx + ix] = g_Matrix[ix * ny + iy]; //read stride and write coalsced
	}
}

__host__ void TransposeOnGPU(float* h_Matrix, float* h_MatrixTranspose, float* refTranspose, int ny, int nx)
{
	float* d_Matrix;
	float* d_MatrixTranspose;
	const int MatrixSizeInBytes = ny * nx * sizeof(float);

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_Matrix, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_MatrixTranspose, MatrixSizeInBytes));

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

	//Matrix Transpose Load by Row (Coalesced Access) and Store by column (Stride Access)
	NaiveRowTranspose << <grid, block >> > (d_Matrix, d_MatrixTranspose, ny, nx);
	cudaDeviceSynchronize();
	HandleCUDAError(cudaMemcpy(h_MatrixTranspose, d_MatrixTranspose, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	//Verify
	VerifyTranspose(h_MatrixTranspose, refTranspose, ny, nx);

	//Zero the computed transpose
	ZeroMatrix(h_MatrixTranspose, ny, nx);
	//Matrix Transpose Load by Column (Stride Access) and Store by Row (Coalesced Access)
	NaiveColTranspose << <grid, block >> > (d_Matrix, d_MatrixTranspose, ny, nx);
	cudaDeviceSynchronize();
	HandleCUDAError(cudaMemcpy(h_MatrixTranspose, d_MatrixTranspose, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	//Verify
	VerifyTranspose(h_MatrixTranspose, refTranspose, ny, nx);

	HandleCUDAError(cudaFree(d_Matrix));
	HandleCUDAError(cudaFree(d_MatrixTranspose));
	HandleCUDAError(cudaDeviceReset());
}
