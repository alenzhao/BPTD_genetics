// utility_gpu.cuh
// function:

#ifndef UTILITY_GPU_CUH
#define UTILITY_GPU_CUH




using namespace std;




typedef struct matrixSize
{
	unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;



void gpu_saxpy_array(float *, float *, int, float);

void gpu_array_multion(int, float *, float *);

void gpu_cal_matrixmul(float *, float *, float *, sMatrixSize &);





#endif

// end of utility_gpu.cuh
