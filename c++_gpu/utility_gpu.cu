// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// private
#include "utility_gpu.cuh"





template <int BLOCK_SIZE> __global__ void
kernel_array_multion(int dimension, float * result, float * input)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// total index (using full width, but not full height)
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;
	long int index = i * blockDim.x * gridDim.x + j;


	if(index < dimension)
	{
		result[index] = result[index] * input[index];
	}


	/* However, share-memory seems not helping, since there are no internal computation with these shared mem
	if(index < dimension)
	{
        // Declaration of the shared memory array
        __shared__ float share_result[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float share_input[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        share_result[ty][tx] = result[index];
        share_input[ty][tx] = input[index];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

		share_result[ty][tx] = share_result[ty][tx] * share_input[ty][tx];

		// Write back, each thread writes one element
		result[index] = share_result[ty][tx];
	}
	*/

	return;
}




// perform Y = a * X + Y, on arrays
void gpu_saxpy_array(float * h_result, float * h_input, int dimension, float coef)
{
	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	// on-device memory config
	float * d_input;
	float * d_result;
	checkCudaErrors(cudaMalloc((void **) &d_input, dimension*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_result, dimension*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_input, h_input, dimension*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_result, h_result, dimension*sizeof(float), cudaMemcpyHostToDevice));

	float al = coef;
	checkCudaErrors(cublasSaxpy(handle, dimension, &al, d_input, 1, d_result, 1));

	// copy result from device to host
	checkCudaErrors(cudaMemcpy(h_result, d_result, dimension*sizeof(float), cudaMemcpyDeviceToHost));

	// clean up memory
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_result));
	// Destroy the handle
	checkCudaErrors(cublasDestroy(handle));

	return;
}





void gpu_array_multion(int dimension, float * h_result, float * h_input)
{
	// on-device memory config
	float * d_result;
	float * d_input;
	checkCudaErrors(cudaMalloc((void **) &d_result, dimension*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_input, dimension*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_result, h_result, dimension*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_input, h_input, dimension*sizeof(float), cudaMemcpyHostToDevice));


	int block_size = 32;
	dim3 threads(block_size, block_size);
	int d_square = int(sqrt(dimension)) + 1;	// re-shape to a square matrix slightly larger than the current array
	dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );

	kernel_array_multion<32><<< grid, threads >>>(dimension, d_result, d_input);


	// copy result from device to host
	checkCudaErrors(cudaMemcpy(h_result, d_result, dimension*sizeof(float), cudaMemcpyDeviceToHost));

	// clean up memory
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_result));

	return;
}





void gpu_cal_matrixmul(float * h_A, float * h_B, float * h_C, sMatrixSize & matrix_size)
{
	//int devID = 0;
	//int block_size = 32;


	// allocate device memory
	float *d_A, *d_B, *d_C;

	unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
	unsigned int mem_size_C = sizeof(float) * size_C;

	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));



	// CUBLAS version 2.0
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;
		cublasHandle_t handle;
		cudaEvent_t start, stop;


		checkCudaErrors(cublasCreate(&handle));


		// Allocate CUDA events that we'll use for timing
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		// Record the start event
		checkCudaErrors(cudaEventRecord(start, NULL));


		//note cublas is column primary!
		//need to transpose the order
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));


		// Record the stop event
		checkCudaErrors(cudaEventRecord(stop, NULL));
		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));
		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


		// copy result from device to host
		checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

		// Destroy the handle
		checkCudaErrors(cublasDestroy(handle));
	}


	// clean up memory
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	return;
}



