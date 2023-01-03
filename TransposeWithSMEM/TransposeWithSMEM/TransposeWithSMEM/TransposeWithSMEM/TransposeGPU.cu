#include "TransposeSMEM.h"
#include "GPUErrors.h"


__global__ void NaiveColTranspose(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	if (ix < nx && iy < ny)
	{
		g_MatrixTranspose[iy * nx + ix] = g_Matrix[ix * ny + iy];
	}
}

//Use of Shared Memory
#define ST_DIMX 16
#define ST_DIMY 16

__global__ void TransposeWithSM(float* g_Matrix, float* g_MatrixTranspose, int ny, int nx)
{
	//Declare static shared memory 
	__shared__ float tile[ST_DIMY][ST_DIMX];

	//Coordinates in original matrix
	unsigned int ix = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int iy = threadIdx.y + (blockIdx.y * blockDim.y);

	//Linear global memory index address in the original matrix
	unsigned int ti = iy * nx + ix;

	//thread index in the transposed block
	unsigned int bidx = threadIdx.x + (threadIdx.y * blockDim.x);
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.x;

	//Coordinates in transpose matrix
	ix = icol + (blockIdx.y * blockDim.y);
	iy = irow + (blockIdx.x * blockDim.x);

	//linear global memory index address in the transpose matrix
	unsigned int to = iy * ny + ix;

	if (ix < nx && iy < ny)
	{
		//Load the data from the original matrix into the tile on the shared memory
		tile[threadIdx.y][threadIdx.x] = g_Matrix[ti];
		__syncthreads();
		g_MatrixTranspose[to] = tile[icol][irow];
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

	//Matrix Transpose Load by Column (Stride Access) and Store by Row (Coalesced Access)
	NaiveColTranspose << <grid, block >> > (d_Matrix, d_MatrixTranspose, ny, nx);
	cudaDeviceSynchronize();
	HandleCUDAError(cudaMemcpy(h_MatrixTranspose, d_MatrixTranspose, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	//Verify
	VerifyTranspose(h_MatrixTranspose, refTranspose, ny, nx);

	//Zero the computed transpose
	ZeroMatrix(h_MatrixTranspose, ny, nx);

	//Shared Memory based Transpose
	TransposeWithSM << <grid, block >> > (d_Matrix, d_MatrixTranspose, ny, nx);
	cudaDeviceSynchronize();
	HandleCUDAError(cudaMemcpy(h_MatrixTranspose, d_MatrixTranspose, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	//Verify
	VerifyTranspose(h_MatrixTranspose, refTranspose, ny, nx);

	HandleCUDAError(cudaFree(d_Matrix));
	HandleCUDAError(cudaFree(d_MatrixTranspose));
	HandleCUDAError(cudaDeviceReset());
}
