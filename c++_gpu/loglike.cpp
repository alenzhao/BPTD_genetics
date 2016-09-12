/*

Hi there, I will calculate the loglikelihood during the training.

*/


#include <iostream>
#include <math.h>       /* sqrt */
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include "utility.h"
#include "global.h"




using namespace std;




// loglike of Gaussian's for a Matrix matrix with given mean value
// NOTE: removed constant terms
float loglike_gaussian_uni(Matrix & data, float mu, float lambda)
{
	float sigma = sqrt(1.0 / lambda);

	float loglike = 0;

	int amount = data.get_dimension1() * data.get_dimension2();
	loglike += amount * ( - log( sigma ) );

	Matrix temp = cal_matrixsubtract(data, mu);
	temp.power(2.0);
	float temp_value = temp.sum();
	loglike += ( - temp_value / ( 2 * pow(sigma, 2.0) ) );

	//==##== collector ==##==
	temp.release();

	return loglike;
}



// loglike of Gaussian's for a Matrix matrix with given mean Matrix Mu
float loglike_gaussian_matrix(Matrix & data, Matrix & Mu, float lambda)
{
	float sigma = sqrt(1.0 / lambda);

	float loglike = 0;

	int amount = data.get_dimension1() * data.get_dimension2();
	loglike += amount * ( - log( sigma ) );

	Matrix temp = cal_matrixsubtract(data, Mu);
	temp.power(2.0);
	float temp_value = temp.sum();
	loglike += ( - temp_value / ( 2 * pow(sigma, 2.0) ) );

	//==##== collector ==##==
	temp.release();

	return loglike;
}



// loglike of Gaussian's for a tensor with given mean tensor
float loglike_gaussian_tensor(Tensor & data, Tensor & Mu, float lambda)
{
	float sigma = sqrt(1.0 / lambda);

	float loglike = 0;

	int amount = data.get_dimension1() * data.get_dimension2() * data.get_dimension3();
	loglike += amount * ( - log( sigma ) );

	Tensor temp = cal_tensorsubtract(data, Mu);
	temp.power(2.0);
	float temp_value = temp.sum();
	loglike += ( - temp_value / ( 2 * pow(sigma, 2.0) ) );

	//==##== collector ==##==
	temp.release();

	return loglike;
}



// loglike of Gaussian's for the data (with markerset indicating the incomplete tensor)
float loglike_gaussian_data(Tensor & data, Tensor & Mu)
{
	float sigma = sqrt(1.0 / alpha);

	Tensor data1 = cal_tensormultiply(data, markerset);
	Tensor Mu1 = cal_tensormultiply(Mu, markerset);

	float loglike = 0;

	float amount = N_element;
	loglike += amount * ( - log( sigma ) );

	Tensor temp = cal_tensorsubtract(data1, Mu1);
	temp.power(2.0);
	float temp_value = temp.sum();
	loglike += ( - temp_value / ( 2 * pow(sigma, 2.0) ) );

	//==##== collector ==##==
	data1.release();
	Mu1.release();
	temp.release();

	return loglike;
}



// loglike of Gamma
float loglike_gamma(float obs, float alpha, float beta)
{
	float loglike = 0;

	loglike += (alpha - 1) * log(obs);
	loglike += (- beta * obs);
	loglike += alpha * log(beta);
	loglike += (- log( tgamma(alpha) ) );

	return loglike;
}







void loglike_cal(float mu0, float lambda0, float alpha0, float beta0)
{
	// global:
	//	X, Y, Y1, U1, V1, T1, Beta, Y2, U2, V2, T2, alpha, K, I, J, S, D1, D2
	//	loglike_total, loglike_data, loglike_Y1, loglike_Y2
	//	loglike_U1, loglike_V1, loglike_T1, loglike_U2, loglike_V2, loglike_T2, loglike_Beta, loglike_alpha

	cout << "calculating the loglike..." << endl;


	float loglike_cumu = 0;
	float loglike;




	// DEBUG
	struct timeval time_start;
	struct timeval time_end;
	double time_diff;





	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==== loglike_data
	Tensor mean = cal_tensoradd_multicoef(1, Y1, 1, Y2);
	loglike = loglike_gaussian_data(Y, mean);
	loglike_data.push_back(loglike);
	loglike_cumu += loglike;
	//==##== collector ==##==
	mean.release();



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==== loglike_Y1
	mean = cal_tensor_innerprod(T1, U1, V1);
	loglike = loglike_gaussian_tensor(Y1, mean, lambda0);
	loglike_Y1.push_back(loglike);
	loglike_cumu += loglike;
	//==##== collector ==##==
	mean.release();



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);



	//==== loglike_Y2
	mean = cal_tensor_innerprod(T2, U2, V2);
	loglike = loglike_gaussian_tensor(Y2, mean, lambda0);
	loglike_Y2.push_back(loglike);
	loglike_cumu += loglike;
	//==##== collector ==##==
	mean.release();


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==== loglike_U1
	Matrix Beta_reshape = op_matrix_rotate(Beta);
	Matrix mean_matrix = cal_matrixmul(X, Beta_reshape);
	loglike = loglike_gaussian_matrix(U1, mean_matrix, lambda0);
	loglike_U1.push_back(loglike);
	loglike_cumu += loglike;
	//==##== collector ==##==
	Beta_reshape.release();
	mean_matrix.release();



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==== loglike_V1
	loglike = loglike_gaussian_uni(V1, mu0, lambda0);
	loglike_V1.push_back(loglike);
	loglike_cumu += loglike;



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==== loglike_T1
	loglike = loglike_gaussian_uni(T1, mu0, lambda0);
	loglike_T1.push_back(loglike);
	loglike_cumu += loglike;



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);



	//==== loglike_U2
	loglike = loglike_gaussian_uni(U2, mu0, lambda0);
	loglike_U2.push_back(loglike);
	loglike_cumu += loglike;



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);



	//==== loglike_V2
	loglike = loglike_gaussian_uni(V2, mu0, lambda0);
	loglike_V2.push_back(loglike);
	loglike_cumu += loglike;



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);



	//==== loglike_T2
	loglike = loglike_gaussian_uni(T2, mu0, lambda0);
	loglike_T2.push_back(loglike);
	loglike_cumu += loglike;




	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);




	//==== loglike_Beta
	loglike = loglike_gaussian_uni(Beta, mu0, lambda0);
	loglike_Beta.push_back(loglike);
	loglike_cumu += loglike;




	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);



	//==== loglike_alpha
	loglike = loglike_gamma(alpha, alpha0, beta0);
	loglike_alpha.push_back(loglike);
	loglike_cumu += loglike;





	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;










	//==== loglike_total
	loglike_total.push_back(loglike_cumu);
	// TODO:
	//np.save("./result/loglike_total", np.array(loglike_total))



	return;
}




