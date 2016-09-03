// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// private
#include "utility_gpu.cuh"






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


