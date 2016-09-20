// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <random>
#include <chrono>		/* sys time */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>


// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

// private
#include "utility_gpu.h"
#include "utility.h"





using namespace std;





//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



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





//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============





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




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============




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




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============




template <int BLOCK_SIZE> __global__ void
kernel_cal_tensorouter(int dimension1, int dimension2, int dimension3, float * fm1, float * fm2, float * result)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;
	long int bz = blockIdx.z;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the real tensor
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;
	long int z = bz;

	// boundary check
	if(i<dimension1 && j<dimension2 && z<dimension3)
	{
		float value = fm1[i * dimension3 + z] * fm2[j * dimension3 + z];
		int index = i * (dimension2 * dimension3) + j * dimension3 + z;
		result[index] = value;
	}

	return;
}





void gpu_cal_tensorouter(float * h_fm1, float * h_fm2, float * h_result, int dimension1, int dimension2, int dimension3)
{
	// allocate device memory
	float *d_fm1, *d_fm2, *d_result;

	checkCudaErrors(cudaMalloc((void **) &d_fm1, dimension1*dimension3*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_fm2, dimension2*dimension3*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_fm1, h_fm1, dimension1*dimension3*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fm2, h_fm2, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &d_result, dimension1*dimension2*dimension3*sizeof(float)));


	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid( (dimension2+threads.x-1)/threads.x, (dimension1+threads.y-1)/threads.y, dimension3 );

	kernel_cal_tensorouter<32><<< grid, threads >>>(dimension1, dimension2, dimension3, d_fm1, d_fm2, d_result);


	// copy result from device to host
	checkCudaErrors(cudaMemcpy(h_result, d_result, dimension1*dimension2*dimension3*sizeof(float), cudaMemcpyDeviceToHost));

	// clean up memory
	checkCudaErrors(cudaFree(d_fm1));
	checkCudaErrors(cudaFree(d_fm2));
	checkCudaErrors(cudaFree(d_result));

	return;
}




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============




template <int BLOCK_SIZE> __global__ void
kernel_cal_tensor_innerprod(int dimension1, int dimension2, int dimension3, int d_factor,
							float * fm1, float * fm2, float * fm3, float * result)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;
	long int bz = blockIdx.z;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// indexing in the real tensor
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;
	long int z = bz;


	/*
	long int id_thread = ty * blockDim.x + tx;


	//==== share-memory loading
    __shared__ float fm1_sub[400];		// NOTE: there are 400 factors now

	int count = id_thread;
	while(count < d_factor)
	{
		fm1_sub[count] = fm1[z * d_factor + count];
		count += 1024;					// NOTE: the total number of threads per block
	}

    // Synchronize to make sure the matrices are loaded
    __syncthreads();
    */


	// boundary check
	if(z<dimension1 && i<dimension2 && j<dimension3)
	{
		float value = 0;
		for(int d=0; d<d_factor; d++)
		{
			value += fm1[z * d_factor + d] * fm2[i * d_factor + d] * fm3[j * d_factor + d];
			//value += fm1_sub[d] * fm2[i * d_factor + d] * fm3[j * d_factor + d];
		}
		int index = z * (dimension2 * dimension3) + i * dimension3 + j;
		result[index] = value;
	}

	return;
}




void gpu_cal_tensor_innerprod(float * h_fm1, float * h_fm2, float * h_fm3,
							float * h_result, int dimension1, int dimension2, int dimension3, int d_factor)
{
	// allocate device memory
	float *d_fm1, *d_fm2, *d_fm3, *d_result;

	checkCudaErrors(cudaMalloc((void **) &d_fm1, dimension1*d_factor*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_fm2, dimension2*d_factor*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_fm3, dimension3*d_factor*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_fm1, h_fm1, dimension1*d_factor*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fm2, h_fm2, dimension2*d_factor*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fm3, h_fm3, dimension3*d_factor*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &d_result, dimension1*dimension2*dimension3*sizeof(float)));


	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid( (dimension3+threads.x-1)/threads.x, (dimension2+threads.y-1)/threads.y, dimension1 );

	kernel_cal_tensor_innerprod<32><<< grid, threads >>>(dimension1, dimension2, dimension3, d_factor, d_fm1, d_fm2, d_fm3, d_result);


	// copy result from device to host
	checkCudaErrors(cudaMemcpy(h_result, d_result, dimension1*dimension2*dimension3*sizeof(float), cudaMemcpyDeviceToHost));

	// clean up memory
	checkCudaErrors(cudaFree(d_fm1));
	checkCudaErrors(cudaFree(d_fm2));
	checkCudaErrors(cudaFree(d_fm3));
	checkCudaErrors(cudaFree(d_result));

	return;
}





//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============





template <int BLOCK_SIZE> __global__ void
kernel_set_with_ref(int dimension1, int dimension2, float * result, float * ref)
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


	if(i<dimension1 && j<dimension2)
	{
		long int index = i * dimension2 + j;
		result[index] = ref[index];
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_set_value(long int dimension1, long int dimension2, float * result, float value)
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


	if(i<dimension1 && j<dimension2)
	{
		long int index = i * dimension2 + j;
		result[index] = value;
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_array_set_value(long int dimension, float * result, float value)
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
		result[index] = value;
	}

	return;
}




void gpu_Gibbs_uniGaussian_fm(
			float lambda_data,
			float lambda_prior,
			int d_feature,
			int d_factor,
			Matrix fm,
			Tensor tensor_reshape,
			Tensor coef_tensor,
//			Tensor & coef_tensor_reshape,				// NOTE: this is useless, but memory-consuming
			Array lambda_list,
			Matrix mu_matrix,
			Tensor mean_tensor)
{
	int dimension1 = tensor_reshape.get_dimension1();
	int dimension2 = tensor_reshape.get_dimension2();
	int dimension3 = tensor_reshape.get_dimension3();




	//==== declare
	float * pointer;




	//==== cached variables
	//
	vector<float> entry_last;
	for(int i=0; i<d_feature; i++)
	{
		entry_last.push_back(0);
	}
	//
	float * d_mean1_array;
	checkCudaErrors(cudaMalloc((void **) &d_mean1_array, dimension1*dimension2*dimension3*sizeof(float)));
	pointer = mean_tensor.get_tensor();
	checkCudaErrors(cudaMemcpy(d_mean1_array, pointer, dimension1*dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
	//
	float * d_data_array;
	checkCudaErrors(cudaMalloc((void **) &d_data_array, dimension1*dimension2*dimension3*sizeof(float)));
	pointer = tensor_reshape.get_tensor();
	checkCudaErrors(cudaMemcpy(d_data_array, pointer, dimension1*dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
	//
	// @@@@ set value 1
	//==============================================================================================================
	// unit vector
	float * d_array_unit;
	checkCudaErrors(cudaMalloc((void **) &d_array_unit, dimension2*dimension3*sizeof(float)));
	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid( (dimension3+threads.x-1)/threads.x, (dimension2+threads.y-1)/threads.y );

	kernel_set_value<32><<< grid, threads >>>(dimension2, dimension3, d_array_unit, 1);
	//==============================================================================================================
	float * d_temp;
	checkCudaErrors(cudaMalloc((void **) &d_temp, dimension2*dimension3*sizeof(float)));
	float * h_temp = (float *)calloc(dimension2*dimension3, sizeof(float));




	//==== main loop here
	for(int d=0; d<d_factor; d++)
	{
		//==== load d_coef_old, and d_coef_new
		// NOTE: for (400, 450, 21150), we can't load them at once, but for others we can; but that makes almost no difference in terms of total speed
		float * d_coef_old;
		float * d_coef_new;
		if(d == 0)
		{
			//== only need coef_new
			checkCudaErrors(cudaMalloc((void **) &d_coef_new, dimension2*dimension3*sizeof(float)));
			pointer = coef_tensor.get_tensor();
			checkCudaErrors(cudaMemcpy(d_coef_new, pointer, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
		}
		else
		{
			//== need both coef_old and coef_new
			checkCudaErrors(cudaMalloc((void **) &d_coef_old, dimension2*dimension3*sizeof(float)));
			pointer = coef_tensor.get_tensor_at(d-1);
			checkCudaErrors(cudaMemcpy(d_coef_old, pointer, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));

			checkCudaErrors(cudaMalloc((void **) &d_coef_new, dimension2*dimension3*sizeof(float)));
			pointer = coef_tensor.get_tensor_at(d);
			checkCudaErrors(cudaMemcpy(d_coef_new, pointer, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
		}


		for(int i=0; i<d_feature; i++)
		{
			float * d_mean1 = d_mean1_array + i * dimension2 * dimension3;
			float * d_data = d_data_array + i * dimension2 * dimension3;


			//== prepare
			float x = fm.get_element(i, d);

			if(d == 0)
			{
				entry_last.at(i) = x;
			}
			else
			{
				float entry_new = fm.get_element(i, d-1);
				// @@@@ Saxpy
				//==============================================================================================================
				// cuBLAS handle
				cublasHandle_t handle;
				checkCudaErrors(cublasCreate(&handle));

				float al = entry_new - entry_last.at(i);
				checkCudaErrors(cublasSaxpy(handle, dimension2*dimension3, &al, d_coef_old, 1, d_mean1, 1));

				// Destroy the handle
				checkCudaErrors(cublasDestroy(handle));
				//==============================================================================================================

				entry_last.at(i) = x;
			}


			float lambda = lambda_list.get_element(d);
			float mu = mu_matrix.get_element(i, d);


			//== compute further
			//float * d_temp;  --> already have this
			// @@@@ misc
			//==============================================================================================================
			int block_size = 32;
			dim3 threads1(block_size, block_size);
			dim3 grid1( (dimension3+threads1.x-1)/threads1.x, (dimension2+threads1.y-1)/threads1.y );

			kernel_set_with_ref<32><<< grid1, threads1 >>>(dimension2, dimension3, d_temp, d_data);
			//==============================================================================================================

			// @@@@ Saxpy
			//==============================================================================================================
			// cuBLAS handle
			cublasHandle_t handle;
			checkCudaErrors(cublasCreate(&handle));

			float al = -1;
			checkCudaErrors(cublasSaxpy(handle, dimension2*dimension3, &al, d_mean1, 1, d_temp, 1));

			// Destroy the handle
			checkCudaErrors(cublasDestroy(handle));
			//==============================================================================================================

			// @@@@ Saxpy
			//==============================================================================================================
			// cuBLAS handle
			//cublasHandle_t handle;
			checkCudaErrors(cublasCreate(&handle));

			//float al = x;
			al = x;
			checkCudaErrors(cublasSaxpy(handle, dimension2*dimension3, &al, d_coef_new, 1, d_temp, 1));

			// Destroy the handle
			checkCudaErrors(cublasDestroy(handle));
			//==============================================================================================================

			// @@@@ multion
			//==============================================================================================================
			//int block_size = 32;
			dim3 threads2(block_size, block_size);
			dim3 grid2( (dimension3+threads2.x-1)/threads2.x, (dimension2+threads2.y-1)/threads2.y );

			kernel_array_multion<32><<< grid2, threads2 >>>(dimension2*dimension3, d_temp, d_coef_new);
			//==============================================================================================================




			// @@@@ dot product with unit vector --> sum (probably there is a better way to do this?)
			//==============================================================================================================
			// cuBLAS handle
			//cublasHandle_t handle;
			checkCudaErrors(cublasCreate(&handle));

			float result = 0;
			checkCudaErrors(cublasSdot(handle, dimension2*dimension3, d_temp, 1, d_array_unit, 1, &result));

			// Destroy the handle
			checkCudaErrors(cublasDestroy(handle));
			//==============================================================================================================
			// @@@@ the CPU counterpart of the above routine (NOTE: NO, the following code is too slow)
			//==============================================================================================================
			// checkCudaErrors(cudaMemcpy(h_temp, d_temp, dimension2*dimension3*sizeof(float), cudaMemcpyDeviceToHost));
			// float result = 0;
			// for(int count=0; count<dimension2*dimension3; count++)
			// {
			// 	result += h_temp[count];
			// }
			//==============================================================================================================
			float temp_value = result;
			//checkCudaErrors(cudaFree(d_temp));




			float mean = lambda_data * temp_value + lambda_prior * mu;
			mean = mean / lambda;

			//== sampler
			float sigma = sqrt(1.0 / lambda);
			//float mu = mean;
			mu = mean;
  			// construct a trivial random generator engine from a time-based seed:
  			unsigned seed = chrono::system_clock::now().time_since_epoch().count();
			default_random_engine generator(seed);
			normal_distribution<double> distribution(mu, sigma);
			float draw = distribution(generator);
			fm.set_element(i, d, draw);
		}


		//==== load release coef_tensor_old, and coef_tensor_new
		if(d == 0)
		{
			checkCudaErrors(cudaFree(d_coef_new));
		}
		else
		{
			checkCudaErrors(cudaFree(d_coef_old));
			checkCudaErrors(cudaFree(d_coef_new));
		}

	}



	//==== release cached variables
	checkCudaErrors(cudaFree(d_mean1_array));
	checkCudaErrors(cudaFree(d_data_array));
	checkCudaErrors(cudaFree(d_array_unit));
	checkCudaErrors(cudaFree(d_temp));
	free(h_temp);


	return;
}



//=============/=============/=============/=============/=============/=============/=============/=============



//==== some sub-routines for "gpu_Gibbs_uniGaussian_fm_parallel"
template <int BLOCK_SIZE> __global__ void
kernel_arrayblock_saxpy_then_multion(long int dimension_total, long int dimension_block, float * matrix_result, float * block_parameter, float * list_coef)
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


	if(index < dimension_total)
	{
		long int block_id = index / dimension_block;
		long int block_pos = index % dimension_block;

		float coef = list_coef[block_id];
		float parameter = block_parameter[block_pos];


		matrix_result[index] = ( matrix_result[index] + coef * parameter ) * parameter;
	}

	return;
}



template <int BLOCK_SIZE> __global__ void
kernel_arrayblock_saxpy(long int dimension_total, long int dimension_block, float * matrix_result, float * block_parameter, float * list_coef)
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


	if(index < dimension_total)
	{
		long int block_id = index / dimension_block;
		long int block_pos = index % dimension_block;

		float coef = list_coef[block_id];
		float parameter = block_parameter[block_pos];

		matrix_result[index] = matrix_result[index] + coef * parameter;
	}

	return;
}




template <int BLOCK_SIZE> __global__ void
kernel_array_deduct(long int dimension, float * result, float * array1, float * array2)
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
		result[index] = array1[index] - array2[index];
	}

	return;
}
//==============================================================





// updated:
//	across factors, we couldn't parallel, since the next factor will depend on the value of the currently being updated factor
//	but across features, we can parallel; we make matrix larger and GPU helpful
void gpu_Gibbs_uniGaussian_fm_parallel(
			float lambda_data,
			float lambda_prior,
			int d_feature,
			int d_factor,
			Matrix fm,
			Tensor tensor_reshape,
			Tensor coef_tensor,
//			Tensor & coef_tensor_reshape,				// NOTE: this is useless, but memory-consuming
			Array lambda_list,
			Matrix mu_matrix,
			Tensor mean_tensor)
{
	int dimension1 = tensor_reshape.get_dimension1();
	int dimension2 = tensor_reshape.get_dimension2();
	int dimension3 = tensor_reshape.get_dimension3();




	//==== declare
	float * pointer;




	//==== cached variables
	//
	float entry_last[d_feature];
	//
	float * d_mean1_array;
	checkCudaErrors(cudaMalloc((void **) &d_mean1_array, dimension1*dimension2*dimension3*sizeof(float)));
	pointer = mean_tensor.get_tensor();
	checkCudaErrors(cudaMemcpy(d_mean1_array, pointer, dimension1*dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
	//
	float * d_data_array;
	checkCudaErrors(cudaMalloc((void **) &d_data_array, dimension1*dimension2*dimension3*sizeof(float)));
	pointer = tensor_reshape.get_tensor();
	checkCudaErrors(cudaMemcpy(d_data_array, pointer, dimension1*dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
	//
	// @@@@ set value 1
	//==============================================================================================================
	// unit vector
	float * d_array_unit;
	checkCudaErrors(cudaMalloc((void **) &d_array_unit, dimension2*dimension3*sizeof(float)));
	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid( (dimension3+threads.x-1)/threads.x, (dimension2+threads.y-1)/threads.y );

	kernel_set_value<32><<< grid, threads >>>(dimension2, dimension3, d_array_unit, 1);
	//==============================================================================================================
	float * d_temp;
	// NOTE: below this becomes the tensor size
	checkCudaErrors(cudaMalloc((void **) &d_temp, dimension1*dimension2*dimension3*sizeof(float)));



	//==== main loop here
	for(int d=0; d<d_factor; d++)
	{

		//==== load d_coef_old, and d_coef_new
		// NOTE: for (400, 450, 21150), we can't load them at once, but for others we can; but that makes almost no difference in terms of total speed
		float * d_coef_old;
		float * d_coef_new;
		if(d == 0)
		{
			//== only need coef_new
			checkCudaErrors(cudaMalloc((void **) &d_coef_new, dimension2*dimension3*sizeof(float)));
			pointer = coef_tensor.get_tensor();
			checkCudaErrors(cudaMemcpy(d_coef_new, pointer, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
		}
		else
		{
			//== need both coef_old and coef_new
			checkCudaErrors(cudaMalloc((void **) &d_coef_old, dimension2*dimension3*sizeof(float)));
			pointer = coef_tensor.get_tensor_at(d-1);
			checkCudaErrors(cudaMemcpy(d_coef_old, pointer, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));

			checkCudaErrors(cudaMalloc((void **) &d_coef_new, dimension2*dimension3*sizeof(float)));
			pointer = coef_tensor.get_tensor_at(d);
			checkCudaErrors(cudaMemcpy(d_coef_new, pointer, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));
		}



		//===========================================================
		//===========================================================
		//===========================================================
		//==== from here we will parallel computing all the features
		//===========================================================
		//===========================================================
		//===========================================================
		//float * d_mean1 = d_mean1_array + i * dimension2 * dimension3;
		//float * d_data = d_data_array + i * dimension2 * dimension3;


		//== prepare
		//float x = fm.get_element(i, d);
		float list_x[d_feature];
		for(int i=0; i<d_feature; i++)
		{
			list_x[i] = fm.get_element(i, d);
		}
		float * d_list_x;
		checkCudaErrors(cudaMalloc((void **) &d_list_x, d_feature*sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_list_x, list_x, d_feature*sizeof(float), cudaMemcpyHostToDevice));


		// if(d == 0)
		// {
		// 	entry_last.at(i) = x;
		// }
		// else
		// {
		// 	float entry_new = fm.get_element(i, d-1);
		// 	// @@@@ Saxpy
		// 	//==============================================================================================================
		// 	// cuBLAS handle
		// 	cublasHandle_t handle;
		// 	checkCudaErrors(cublasCreate(&handle));

		// 	float al = entry_new - entry_last.at(i);
		// 	checkCudaErrors(cublasSaxpy(handle, dimension2*dimension3, &al, d_coef_old, 1, d_mean1, 1));

		// 	// Destroy the handle
		// 	checkCudaErrors(cublasDestroy(handle));
		// 	//==============================================================================================================

		// 	entry_last.at(i) = x;
		// }
		if(d == 0)
		{
			for(int i=0; i<d_feature; i++)
			{
				entry_last[i] = list_x[i];
			}
		}
		else
		{
			float entry_new[d_feature];
			float entry_delta[d_feature];
			for(int i=0; i<d_feature; i++)
			{
				entry_new[i] = fm.get_element(i, d-1);
				entry_delta[i] = entry_new[i] - entry_last[i];
			}

			// @@@@ Saxpy block kernel
			//==============================================================================================================
			float * d_entry_delta;
			checkCudaErrors(cudaMalloc((void **) &d_entry_delta, d_feature*sizeof(float)));
			checkCudaErrors(cudaMemcpy(d_entry_delta, entry_delta, d_feature*sizeof(float), cudaMemcpyHostToDevice));

			int block_size = 32;
			dim3 threads(block_size, block_size);
			int d_square = int(sqrt(dimension1*dimension2*dimension3)) + 1;	// re-shape to a square matrix slightly larger than the current array
			dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );

			kernel_arrayblock_saxpy<32><<< grid, threads >>>(dimension1*dimension2*dimension3, dimension2*dimension3, d_mean1_array, d_coef_old, d_entry_delta);

			checkCudaErrors(cudaFree(d_entry_delta));
			//==============================================================================================================

			for(int i=0; i<d_feature; i++)
			{
				entry_last[i] = list_x[i];
			}
		}



		//float lambda = lambda_list.get_element(d);
		//float mu = mu_matrix.get_element(i, d);



		//== compute further
		//float * d_temp;  --> already have this
		// @@@@ misc
		// //==============================================================================================================
		// int block_size = 32;
		// dim3 threads(block_size, block_size);
		// dim3 grid( (dimension3+threads.x-1)/threads.x, (dimension2+threads.y-1)/threads.y );

		// kernel_set_with_ref<32><<< grid, threads >>>(dimension2, dimension3, d_temp, d_data);
		// //==============================================================================================================
		// // @@@@ Saxpy
		// //==============================================================================================================
		// // cuBLAS handle
		// cublasHandle_t handle;
		// checkCudaErrors(cublasCreate(&handle));

		// float al = -1;
		// checkCudaErrors(cublasSaxpy(handle, dimension2*dimension3, &al, d_mean1, 1, d_temp, 1));

		// // Destroy the handle
		// checkCudaErrors(cublasDestroy(handle));
		// //==============================================================================================================
		// @@@@ deduct
		//==============================================================================================================
		int block_size = 32;
		dim3 threads(block_size, block_size);
		int d_square = int(sqrt(dimension1*dimension2*dimension3)) + 1;	// re-shape to a square matrix slightly larger than the current array
		dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );

		kernel_array_deduct<32><<< grid, threads >>>(dimension1*dimension2*dimension3, d_temp, d_data_array, d_mean1_array);
		//==============================================================================================================





		// @@@@ Saxpy
		// //==============================================================================================================
		// // cuBLAS handle
		// //cublasHandle_t handle;
		// checkCudaErrors(cublasCreate(&handle));

		// //float al = x;
		// al = x;
		// checkCudaErrors(cublasSaxpy(handle, dimension2*dimension3, &al, d_coef_new, 1, d_temp, 1));

		// // Destroy the handle
		// checkCudaErrors(cublasDestroy(handle));
		// //==============================================================================================================
		// // @@@@ multion
		// //==============================================================================================================
		// //int block_size = 32;
		// dim3 threads2(block_size, block_size);
		// dim3 grid2( (dimension3+threads.x-1)/threads.x, (dimension2+threads.y-1)/threads.y );

		// kernel_array_multion<32><<< grid2, threads2 >>>(dimension2*dimension3, d_temp, d_coef_new);
		// //==============================================================================================================
		// @@@@ block saxpy, and then multion
		//==============================================================================================================
		/*
		int block_size = 32;
		dim3 threads(block_size, block_size);
		int d_square = int(sqrt(dimension1*dimension2*dimension3)) + 1;	// re-shape to a square matrix slightly larger than the current array
		dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );
		*/

		kernel_arrayblock_saxpy_then_multion<32><<< grid, threads >>>(dimension1*dimension2*dimension3, dimension2*dimension3, d_temp, d_coef_new, d_list_x);
		//==============================================================================================================




		// @@@@ dot product with unit vector --> sum (probably there is a better way to do this?)
		//==============================================================================================================
		// // cuBLAS handle
		// //cublasHandle_t handle;
		// checkCudaErrors(cublasCreate(&handle));

		// float result = 0;
		// checkCudaErrors(cublasSdot(handle, dimension2*dimension3, d_temp, 1, d_array_unit, 1, &result));

		// // Destroy the handle
		// checkCudaErrors(cublasDestroy(handle));
		//==============================================================================================================
		// grouped sum --> matrix-array multiplication?
		float list_temp_value[d_feature];
		// @@@@ dot product with unit vector --> sum (probably there is a better way to do this?)
		//==============================================================================================================
		float * d_list_temp_value;
		checkCudaErrors(cudaMalloc((void **) &d_list_temp_value, d_feature*sizeof(float)));


		const float coef_alpha = 1.0f;
		const float coef_beta  = 0.0f;

		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));

		//note cublas is column primary!
		//need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, d_feature, dimension2*dimension3, &coef_alpha, d_array_unit, 1, d_temp, dimension2*dimension3, &coef_beta, d_list_temp_value, 1));


		checkCudaErrors(cudaMemcpy(list_temp_value, d_list_temp_value, d_feature*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_list_temp_value));

		// Destroy the handle
		checkCudaErrors(cublasDestroy(handle));
		//==============================================================================================================





		// now perform the iterations (random sampling, from CPU)
		// NOTE: later might move onto cuRAND
		for(int i=0; i<d_feature; i++)
		{
			float lambda = lambda_list.get_element(d);
			float mu = mu_matrix.get_element(i, d);


			float temp_value = list_temp_value[i];
			//checkCudaErrors(cudaFree(d_temp));

			float mean = lambda_data * temp_value + lambda_prior * mu;
			mean = mean / lambda;

			//== sampler
			float sigma = sqrt(1.0 / lambda);
			//float mu = mean;
			mu = mean;
  			// construct a trivial random generator engine from a time-based seed:
  			unsigned seed = chrono::system_clock::now().time_since_epoch().count();
			default_random_engine generator(seed);
			normal_distribution<double> distribution(mu, sigma);
			float draw = distribution(generator);
			fm.set_element(i, d, draw);
		}



		checkCudaErrors(cudaFree(d_list_x));


		//===========================================================
		//===========================================================
		//===========================================================
		//==== end
		//===========================================================
		//===========================================================
		//===========================================================


		//==== load release coef_tensor_old, and coef_tensor_new
		if(d == 0)
		{
			checkCudaErrors(cudaFree(d_coef_new));
		}
		else
		{
			checkCudaErrors(cudaFree(d_coef_old));
			checkCudaErrors(cudaFree(d_coef_new));
		}

	}



	//==== release cached variables
	checkCudaErrors(cudaFree(d_mean1_array));
	checkCudaErrors(cudaFree(d_data_array));
	checkCudaErrors(cudaFree(d_array_unit));
	checkCudaErrors(cudaFree(d_temp));



	return;
}




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============




void gpu_Gibbs_uniGaussian_Beta(
			float lambda_data,
			float lambda_prior,
			int d_factor,
			int d_snp,

			Matrix fm,
			Matrix U1_reshape,
			Matrix X_reshape,
			Array lambda_list,
			Matrix mean_matrix)
{
	int dimension1 = U1_reshape.get_dimension1();
	int dimension2 = U1_reshape.get_dimension2();



	//==== declare
	float * pointer;
	int d_feature = d_factor;				// transform the view from SNP to just a FM
	//int d_factor = d_snp;
	d_factor = d_snp;



	//==== cached variables
	//
	float entry_last[d_feature];
	//
	float * d_mean1_array;
	checkCudaErrors(cudaMalloc((void **) &d_mean1_array, dimension1*dimension2*sizeof(float)));
	pointer = mean_matrix.get_matrix();
	checkCudaErrors(cudaMemcpy(d_mean1_array, pointer, dimension1*dimension2*sizeof(float), cudaMemcpyHostToDevice));
	//
	float * d_data_array;
	checkCudaErrors(cudaMalloc((void **) &d_data_array, dimension1*dimension2*sizeof(float)));
	pointer = U1_reshape.get_matrix();
	checkCudaErrors(cudaMemcpy(d_data_array, pointer, dimension1*dimension2*sizeof(float), cudaMemcpyHostToDevice));
	//
	// @@@@ set value 1
	//==============================================================================================================
	// unit vector
	float * d_array_unit;
	checkCudaErrors(cudaMalloc((void **) &d_array_unit, dimension2*sizeof(float)));
	int block_size = 32;
	dim3 threads(block_size, block_size);
	int d_square = int(sqrt(dimension2)) + 1;	// re-shape to a square matrix slightly larger than the current array
	dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );

	kernel_array_set_value<32><<< grid, threads >>>(dimension2, d_array_unit, 1);
	//==============================================================================================================
	//
	float * d_temp;
	// NOTE: below this becomes the matrix size
	checkCudaErrors(cudaMalloc((void **) &d_temp, dimension1*dimension2*sizeof(float)));
	//
	float * d_coef_matrix;
	checkCudaErrors(cudaMalloc((void **) &d_coef_matrix, d_factor*dimension2*sizeof(float)));
	pointer = X_reshape.get_matrix();
	checkCudaErrors(cudaMemcpy(d_coef_matrix, pointer, d_factor*dimension2*sizeof(float), cudaMemcpyHostToDevice));




	//==== main loop here
	for(int d=0; d<d_factor; d++)
	{

		//==== prepare d_coef_old, and d_coef_new
		float * d_coef_old;
		float * d_coef_new;
		if(d == 0)
		{
			d_coef_new = d_coef_matrix;
		}
		else
		{
			d_coef_old = d_coef_matrix + (d-1)*dimension2;
			d_coef_new = d_coef_matrix + d*dimension2;
		}


		//===========================================================
		//===========================================================
		//===========================================================
		//==== from here we will parallel computing all the features
		//===========================================================
		//===========================================================
		//===========================================================
		//== prepare
		float list_x[d_feature];
		for(int i=0; i<d_feature; i++)
		{
			list_x[i] = fm.get_element(i, d);
		}
		float * d_list_x;
		checkCudaErrors(cudaMalloc((void **) &d_list_x, d_feature*sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_list_x, list_x, d_feature*sizeof(float), cudaMemcpyHostToDevice));


		if(d == 0)
		{
			for(int i=0; i<d_feature; i++)
			{
				entry_last[i] = list_x[i];
			}
		}
		else
		{
			float entry_new[d_feature];
			float entry_delta[d_feature];
			for(int i=0; i<d_feature; i++)
			{
				entry_new[i] = fm.get_element(i, d-1);
				entry_delta[i] = entry_new[i] - entry_last[i];
			}

			// @@@@ Saxpy block kernel
			//==============================================================================================================
			float * d_entry_delta;
			checkCudaErrors(cudaMalloc((void **) &d_entry_delta, d_feature*sizeof(float)));
			checkCudaErrors(cudaMemcpy(d_entry_delta, entry_delta, d_feature*sizeof(float), cudaMemcpyHostToDevice));

			int block_size = 32;
			dim3 threads(block_size, block_size);
			int d_square = int(sqrt(dimension1*dimension2)) + 1;	// re-shape to a square matrix slightly larger than the current array
			dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );

			kernel_arrayblock_saxpy<32><<< grid, threads >>>(dimension1*dimension2, dimension2, d_mean1_array, d_coef_old, d_entry_delta);

			checkCudaErrors(cudaFree(d_entry_delta));
			//==============================================================================================================

			for(int i=0; i<d_feature; i++)
			{
				entry_last[i] = list_x[i];
			}
		}


		// @@@@ deduct
		//==============================================================================================================
		int block_size = 32;
		dim3 threads(block_size, block_size);
		int d_square = int(sqrt(dimension1*dimension2)) + 1;	// re-shape to a square matrix slightly larger than the current array
		dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );

		kernel_array_deduct<32><<< grid, threads >>>(dimension1*dimension2, d_temp, d_data_array, d_mean1_array);
		//==============================================================================================================



		// @@@@ block saxpy, and then multion
		//==============================================================================================================
		/*
		int block_size = 32;
		dim3 threads(block_size, block_size);
		int d_square = int(sqrt(dimension1*dimension2*dimension3)) + 1;	// re-shape to a square matrix slightly larger than the current array
		dim3 grid( (d_square+threads.x-1)/threads.x, (d_square+threads.y-1)/threads.y );
		*/

		kernel_arrayblock_saxpy_then_multion<32><<< grid, threads >>>(dimension1*dimension2, dimension2, d_temp, d_coef_new, d_list_x);
		//==============================================================================================================



		// grouped sum --> matrix-array multiplication?
		float list_temp_value[d_feature];
		// @@@@ dot product with unit vector --> sum (probably there is a better way to do this?)
		//==============================================================================================================
		float * d_list_temp_value;
		checkCudaErrors(cudaMalloc((void **) &d_list_temp_value, d_feature*sizeof(float)));


		const float coef_alpha = 1.0f;
		const float coef_beta  = 0.0f;

		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));

		//note cublas is column primary!
		//need to transpose the order
		//checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, d_feature, dimension2, &coef_alpha, d_array_unit, 1, d_temp, dimension2, &coef_beta, d_list_temp_value, 1));


		checkCudaErrors(cudaMemcpy(list_temp_value, d_list_temp_value, d_feature*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_list_temp_value));

		// Destroy the handle
		checkCudaErrors(cublasDestroy(handle));
		//==============================================================================================================




		// now perform the iterations (random sampling, from CPU)
		// NOTE: later might move onto cuRAND
		for(int i=0; i<d_feature; i++)
		{
			float lambda = lambda_list.get_element(d);
			//float mu = mu_matrix.get_element(i, d);
			float mu = 0;


			float temp_value = list_temp_value[i];
			//checkCudaErrors(cudaFree(d_temp));

			float mean = lambda_data * temp_value + lambda_prior * mu;
			mean = mean / lambda;

			//== sampler
			float sigma = sqrt(1.0 / lambda);
			//float mu = mean;
			mu = mean;
  			// construct a trivial random generator engine from a time-based seed:
  			unsigned seed = chrono::system_clock::now().time_since_epoch().count();
			default_random_engine generator(seed);
			normal_distribution<double> distribution(mu, sigma);
			float draw = distribution(generator);
			fm.set_element(i, d, draw);
		}


		checkCudaErrors(cudaFree(d_list_x));
		//===========================================================
		//===========================================================
		//===========================================================
		//==== end
		//===========================================================
		//===========================================================
		//===========================================================

	}



	//==== release cached variables
	checkCudaErrors(cudaFree(d_mean1_array));
	checkCudaErrors(cudaFree(d_data_array));
	checkCudaErrors(cudaFree(d_array_unit));
	checkCudaErrors(cudaFree(d_temp));
	checkCudaErrors(cudaFree(d_coef_matrix));


	return;
}





//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============





template <int BLOCK_SIZE> __global__ void
kernel_matrix_square(long int dimension1, long int dimension2, float * matrix)
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


	if(i<dimension1 && j<dimension2)
	{
		long int index = i * dimension2 + j;
		float value = matrix[index];
		matrix[index] = value * value;
	}

	return;
}





void gpu_tensor_sosq_at(float * array_result, Tensor tensor)
{
	int dimension1 = tensor.get_dimension1();
	int dimension2 = tensor.get_dimension2();
	int dimension3 = tensor.get_dimension3();
	// @@@@ set value 1
	//==============================================================================================================
	// unit vector
	float * d_array_unit;
	checkCudaErrors(cudaMalloc((void **) &d_array_unit, dimension2*dimension3*sizeof(float)));
	int block_size = 32;
	dim3 threads(block_size, block_size);
	dim3 grid( (dimension3+threads.x-1)/threads.x, (dimension2+threads.y-1)/threads.y );

	kernel_set_value<32><<< grid, threads >>>(dimension2, dimension3, d_array_unit, 1);
	//==============================================================================================================
	float * d_pointer;
	checkCudaErrors(cudaMalloc((void **) &d_pointer, dimension2*dimension3*sizeof(float)));



	for(int i=0; i<dimension1; i++)
	{
		float * pointer = tensor.get_tensor_at(i);
		checkCudaErrors(cudaMemcpy(d_pointer, pointer, dimension2*dimension3*sizeof(float), cudaMemcpyHostToDevice));


		// @@@@ matrix square
		//==============================================================================================================
		int block_size = 32;
		dim3 threads2(block_size, block_size);
		dim3 grid2( (dimension3+threads.x-1)/threads.x, (dimension2+threads.y-1)/threads.y );

		kernel_matrix_square<32><<< grid2, threads2 >>>(dimension2, dimension3, d_pointer);
		//==============================================================================================================

		// @@@@ dot product with unit vector --> sum (probably there is a better way to do this?)
		//==============================================================================================================
		// cuBLAS handle
		cublasHandle_t handle;
		checkCudaErrors(cublasCreate(&handle));

		float result = 0;
		checkCudaErrors(cublasSdot(handle, dimension2*dimension3, d_pointer, 1, d_array_unit, 1, &result));

		// Destroy the handle
		checkCudaErrors(cublasDestroy(handle));
		//==============================================================================================================


		array_result[i] = result;
	}


	//==== release cached variables
	checkCudaErrors(cudaFree(d_array_unit));
	checkCudaErrors(cudaFree(d_pointer));



	return;
}



