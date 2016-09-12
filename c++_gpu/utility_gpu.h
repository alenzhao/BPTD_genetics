// utility_gpu.h
// function: non-cuda routines, as a interface with CPU code

#ifndef UTILITY_GPU_H
#define UTILITY_GPU_H



#include "utility.h"




using namespace std;




typedef struct matrixSize
{
	unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
}sMatrixSize;



void gpu_saxpy_array(float *, float *, int, float);

void gpu_array_multion(int, float *, float *);

void gpu_cal_matrixmul(float *, float *, float *, sMatrixSize &);

void gpu_cal_tensorouter(float *, float *, float *, int, int, int);

void gpu_cal_tensor_innerprod(float *, float *, float *, float *, int, int, int, int);

void gpu_Gibbs_uniGaussian_fm(float, float, int, int, Matrix, Tensor, Tensor, Array, Matrix, Tensor);
void gpu_Gibbs_uniGaussian_fm_parallel(float, float, int, int, Matrix, Tensor, Tensor, Array, Matrix, Tensor);
void gpu_Gibbs_uniGaussian_Beta(float, float, int, int, Matrix, Matrix, Matrix, Array, Matrix);


void gpu_tensor_sosq_at(float *, Tensor);




#endif

// end of utility_gpu.h
