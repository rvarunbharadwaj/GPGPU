#include "MatrixMult.h"
#include "GPUErrors.h"

__global__ void NaiveMult(float* g_A, float* g_B, float* g_C, const int ny, const int nx)
{
	int row = threadIdx.y + (blockIdx.y * blockDim.y);
	int col = threadIdx.x + (blockIdx.x * blockDim.x);
	float fSum = 0.0f;
	if (row < ny && col < nx)
	{
		for (int k = 0; k < nx; k++)
		{
			fSum += g_A[row * nx + k] * g_B[k * nx + col];//CGMA = 2/8 //2 additions and two floats fetched
		}
		g_C[row * nx + col] = fSum;
	}
}

//Tiled Kernel
#define TILE_WIDTH 32 //tile width is equal to block width
__global__ void TiledMult(float* g_A, float* g_B, float* g_C, const int Width)
{
	//Define a static 2D array on the shared memory of size TILE_WIDTH * TILE_WIDTH to store the elements of the matrix A// data in shared memory is arranged as banks(only for static shared memeory not applicable for dynamic)
	__shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
	//Define a static 2D array on the shared memory of size TILE_WIDTH * TILE_WIDTH to store the elements of the matrix B
	__shared__ float Bds[TILE_WIDTH][TILE_WIDTH]; //Scope of shared memeory is a block //All threads of a block are created to the same Ads and BDs

	//Write code to store locally the thread and block indices
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//Compute the row and column indices of the C Element to store the dot product result
	int row = ty + (by * TILE_WIDTH);
	int col = tx + (bx * TILE_WIDTH);

	//Loop over the A and B tiles to compute the C Element
	float cValue = 0.0f;
	for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) //compared to naive implementation we only do in phase
	{
		//Load A and B tiles into shared memory collaboratively
		Ads[ty][tx] = g_A[row * Width + ph * TILE_WIDTH + tx]; //row*width beginning address of row, ph*tile_width will give beginning of tile and thread will give index in the tiles
		Bds[ty][tx] = g_B[(ph * TILE_WIDTH + ty) * Width + col];
		//Wait for threads of the block (TILE_WIDTH) to complete the loading to the shared memeory in tiles
		__syncthreads();

		//Perform the partial dot product in the phase
		for (int k = 0; k < TILE_WIDTH; k++)
		{
			//for same k we have 16 ty values hence 16 bank conflicts per iteration
			cValue += Ads[ty][k] * Bds[k][tx];//shared mem of A is accessed in a coalased way //CGMa = 2/(8/16) = 4
		}
		//Wait for all threads to complete partial dot product in a phase
		__syncthreads();
	}
	g_C[row * Width + col] = cValue;
	
}

__host__ void gpuMultHelper(float* h_A, float* h_B, float* h_C, float* h_C_Tile,float* ref, const int ny, const int nx)
{
	float* d_A, * d_B, * d_C;
	const int MatrixSizeInBytes = ny * nx * sizeof(float);
	chrono::time_point<high_resolution_clock> start, end;
	double computeTime{};

	//Allocate device memory on the global memory
	HandleCUDAError(cudaMalloc((void**)&d_A, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_B, MatrixSizeInBytes));
	HandleCUDAError(cudaMalloc((void**)&d_C, MatrixSizeInBytes));

	//transfer data from CPU Memory to GPU Memory
	HandleCUDAError(cudaMemcpy(d_A, h_A, MatrixSizeInBytes, cudaMemcpyHostToDevice));
	HandleCUDAError(cudaMemcpy(d_B, h_B, MatrixSizeInBytes, cudaMemcpyHostToDevice));

	//Kernel Invoke Parameters - 2D Grid and 2D Blocks
	int dimx = 32;
	int dimy = 32;

	dim3 block(dimy, dimx);
	dim3 grid((ny + block.y - 1) / block.y, (nx + block.x - 1) / block.x);

	cout << "\t2D Grid Dimension" << endl;
	cout << "\tNumber of Blocks along X dimension: " << grid.x << endl;
	cout << "\tNumber of Blocks along Y dimension: " << grid.y << endl;
	cout << "\t2D Block Dimension" << endl;
	cout << "\tNumber of threads along X dimension: " << block.x << endl;
	cout << "\tNumber of threads along Y dimension: " << block.y << endl;

	//Executing Naive Multiplication
	start = high_resolution_clock::now();
	NaiveMult << <grid, block >> > (d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();
	end = high_resolution_clock::now();
	auto elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "Naive Multiplication: GPU Execution time: " << computeTime << " usecs" << endl;

	HandleCUDAError(cudaMemcpy(h_C, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));

	MatrixMultVerification(ref, h_C, ny, nx);
	//Release the device memory of the C Matrix (Product matrix)
	HandleCUDAError(cudaFree(d_C));
	
	//Executing the Tiled Matrix Multiplication
	//Reallocate the device memory of the C Matrix
	HandleCUDAError(cudaMalloc((void**)&d_C, MatrixSizeInBytes));
	start = high_resolution_clock::now();
	TiledMult << <grid, block >> > (d_A, d_B, d_C, nx);//using static shared memory
	cudaDeviceSynchronize();
	end = high_resolution_clock::now();
	elasped_seconds = end - start;
	computeTime = duration_cast<microseconds>(elasped_seconds).count();
	cout << "Tiled Multiplication: GPU Execution time: " << computeTime << " usecs" << endl;

	HandleCUDAError(cudaMemcpy(h_C_Tile, d_C, MatrixSizeInBytes, cudaMemcpyDeviceToHost));
	MatrixMultVerification(ref, h_C_Tile, ny, nx);
	HandleCUDAError(cudaFree(d_C));
	
	HandleCUDAError(cudaFree(d_A));
	HandleCUDAError(cudaFree(d_B));
	HandleCUDAError(cudaDeviceReset());
}