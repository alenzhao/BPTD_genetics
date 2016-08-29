// global.h
// function: global variables shared among routines

#ifndef GLOBAL_H
#define GLOBAL_H



#include <vector>
#include "utility.h"			// Array, Matrix, and Tensor





using namespace std;





extern Matrix X;
extern Tensor Y;
extern Tensor markerset;
extern Tensor Y1;
extern Matrix U1;
extern Matrix V1;
extern Matrix T1;
extern Matrix Beta;
extern Tensor Y2;
extern Matrix U2;
extern Matrix V2;
extern Matrix T2;
extern float alpha;
// NOTE: the following are the real scenario for chr22 and brain tensor
extern int K;
extern int I;
extern int J;
extern int S;
extern int D1;
extern int D2;

extern int N_element;






extern vector<float> loglike_total;
extern vector<float> loglike_data;
extern vector<float> loglike_Y1;
extern vector<float> loglike_Y2;
extern vector<float> loglike_U1;
extern vector<float> loglike_V1;
extern vector<float> loglike_T1;
extern vector<float> loglike_U2;
extern vector<float> loglike_V2;
extern vector<float> loglike_T2;
extern vector<float> loglike_Beta;
extern vector<float> loglike_alpha;






#endif

// end of global.h
