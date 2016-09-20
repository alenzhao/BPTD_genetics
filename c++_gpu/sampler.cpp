/*

Aha, here comes the Gibbs sampling program

*/


// system
#include <iostream>
#include <math.h>       /* sqrt */
#include <random>
#include <chrono>		/* sys time */
#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

// private
#include "global.h"
#include "utility.h"
#include "utility_gpu.h"





using namespace std;




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



//// sampling two sub-tensors, Y1 and Y2
void sampler_subT(float lambda)
{
	cout << "sampling sub-tensors..." << endl;
	//global Y, Y1, Y2, U1, V1, T1, U2, V2, T2, I, J, K, alpha, markerset



	// DEBUG
	struct timeval time_start;
	struct timeval time_end;
	double time_diff;



	//==== sample Y1
	cout << "sample Y1..." << endl;
	//== pre-cal


	//==== timer starts
	gettimeofday(&time_start, NULL);



	Tensor mean1 = cal_tensor_innerprod(T1, U1, V1);



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);



	//==== timer starts
	gettimeofday(&time_start, NULL);



	Tensor mean2 = cal_tensorsubtract(Y, Y2);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);

	//==== timer starts
	gettimeofday(&time_start, NULL);



	//== combine two Gaussian's
	float lambda1 = lambda;
	float lambda2 = alpha;
	float lambda_new = lambda1 + lambda2;


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);

	//==== timer starts
	gettimeofday(&time_start, NULL);


	Tensor mean_new = cal_tensoradd_multicoef( (lambda1 / lambda_new), mean1, (lambda2 / lambda_new), mean2 );


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== sample
	float sigma = sqrt(1.0 / lambda_new);
	float sigma1 = sqrt(1.0 / lambda1);
	for(int k=0; k<K; k++)
	{
		for(int i=0; i<I; i++)
		{
			for(int j=0; j<J; j++)
			{
				if(markerset.get_element(k,i,j) == 1)
				{
					float mu = mean_new.get_element(k,i,j);

					// construct a trivial random generator engine from a time-based seed:
  					unsigned seed = chrono::system_clock::now().time_since_epoch().count();
					default_random_engine generator(seed);
					normal_distribution<double> distribution(mu, sigma);
					float draw = distribution(generator);

					Y1.set_element(k,i,j, draw);
				}
				else
				{
					float mu = mean1.get_element(k,i,j);

  					// construct a trivial random generator engine from a time-based seed:
  					unsigned seed = chrono::system_clock::now().time_since_epoch().count();
					default_random_engine generator(seed);
					normal_distribution<double> distribution(mu, sigma1);
					float draw = distribution(generator);

					Y1.set_element(k,i,j, draw);
				}
			}
		}
	}



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);




	//==##== collector ==##==
	mean1.release();
	mean2.release();
	mean_new.release();












	//==== sample Y2
	cout << "sample Y2..." << endl;
	//== pre-cal
	//Tensor mean1 = cal_tensor_innerprod(T2, U2, V2);
	mean1 = cal_tensor_innerprod(T2, U2, V2);
	//Tensor mean2 = cal_tensorsubtract(Y, Y1);
	mean2 = cal_tensorsubtract(Y, Y1);
	//== combine two Gaussian's
	//float lambda1 = lambda;
	lambda1 = lambda;
	//float lambda2 = alpha;
	lambda2 = alpha;
	//float lambda_new = lambda1 + lambda2;
	lambda_new = lambda1 + lambda2;
	//Tensor mean_new = cal_tensoradd_multicoef( (lambda1 / lambda_new), mean1, (lambda2 / lambda_new), mean2 );
	mean_new = cal_tensoradd_multicoef( (lambda1 / lambda_new), mean1, (lambda2 / lambda_new), mean2 );
	//== sample
	//float sigma = sqrt(1.0 / lambda_new);
	sigma = sqrt(1.0 / lambda_new);
	//float sigma1 = sqrt(1.0 / lambda1);
	sigma1 = sqrt(1.0 / lambda1);
	for(int k=0; k<K; k++)
	{
		for(int i=0; i<I; i++)
		{
			for(int j=0; j<J; j++)
			{
				if(markerset.get_element(k,i,j) == 1)
				{
					float mu = mean_new.get_element(k,i,j);

  					// construct a trivial random generator engine from a time-based seed:
  					unsigned seed = chrono::system_clock::now().time_since_epoch().count();
					default_random_engine generator(seed);
					normal_distribution<double> distribution(mu, sigma);
					float draw = distribution(generator);

					Y2.set_element(k,i,j, draw);
				}
				else
				{
					float mu = mean1.get_element(k,i,j);

  					// construct a trivial random generator engine from a time-based seed:
  					unsigned seed = chrono::system_clock::now().time_since_epoch().count();
					default_random_engine generator(seed);
					normal_distribution<double> distribution(mu, sigma1);
					float draw = distribution(generator);

					Y2.set_element(k,i,j, draw);
				}
			}
		}
	}

	//==##== collector ==##==
	mean1.release();
	mean2.release();
	mean_new.release();


	return;
}



//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



// the following samples the uniGaussian factor matrix (uni-variance for both data and prior)
void Gibbs_uniGaussian_fm_parallel(
			float lambda_data,
			float lambda_prior,
			int d_feature,
			int d_factor,
			Matrix & fm,
			Tensor & tensor_reshape,
			Tensor & coef_tensor,
//			Tensor & coef_tensor_reshape,				// NOTE: this is useless, but memory-consuming
			Array & lambda_list,
			Matrix & mu_matrix,
			Tensor & mean_tensor)
{

	gpu_Gibbs_uniGaussian_fm_parallel(lambda_data, lambda_prior, d_feature, d_factor, fm, tensor_reshape, coef_tensor, lambda_list, mu_matrix, mean_tensor);

	return;
}


// the following samples the uniGaussian factor matrix (uni-variance for both data and prior)
void Gibbs_uniGaussian_fm(
			float lambda_data,
			float lambda_prior,
			int d_feature,
			int d_factor,
			Matrix & fm,
			Tensor & tensor_reshape,
			Tensor & coef_tensor,
//			Tensor & coef_tensor_reshape,				// NOTE: this is useless, but memory-consuming
			Array & lambda_list,
			Matrix & mu_matrix,
			Tensor & mean_tensor)
{


	gpu_Gibbs_uniGaussian_fm(lambda_data, lambda_prior, d_feature, d_factor, fm, tensor_reshape, coef_tensor, lambda_list, mu_matrix, mean_tensor);


	// code below: transform to GPU in-memory computing
	/*
	// DEBUG
	struct timeval time_start;
	struct timeval time_end;
	double time_diff;




	for(int i=0; i<d_feature; i++)
	{

		//==== timer starts
		gettimeofday(&time_start, NULL);



		float entry_last = 0;
		Matrix mean1 = extract_matrix_from_tensor(mean_tensor, i);

		Matrix data = extract_matrix_from_tensor(tensor_reshape, i);




		for(int d=0; d<d_factor; d++)
		{



			//== prepare
			float x = fm.get_element(i, d);
			//Matrix data = extract_matrix_from_tensor(tensor_reshape, i);


			// DEBUG
			struct timeval time_start1;
			struct timeval time_end1;
			double time_diff1;



			//==== timer starts
			gettimeofday(&time_start1, NULL);



			//Array array = extract_array_from_matrix(fm, i);
			//Matrix mean1 = cal_tensordot(array, coef_tensor_reshape);
			if(d == 0)
			{
				entry_last = x;
			}
			else
			{
				float entry_new = fm.get_element(i, d-1);
				Matrix matrix = extract_matrix_from_tensor(coef_tensor, d-1);
				cal_matrixaddon_multicoef( mean1, matrix, entry_new - entry_last );
				entry_last = x;

				//==##== collector ==##==
				matrix.release();
			}



			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;

			//==== timer starts
			gettimeofday(&time_start1, NULL);




			Matrix coef = extract_matrix_from_tensor(coef_tensor, d);

			float lambda = lambda_list.get_element(d);
			float mu = mu_matrix.get_element(i, d);


			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;


			//==== timer starts
			gettimeofday(&time_start1, NULL);

			//== compute further
			Matrix temp = cal_matrixsubtract(data, mean1);

			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;

			//==== timer starts
			gettimeofday(&time_start1, NULL);


			cal_matrixaddon_multicoef(temp, coef, x);


			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;


			//==== timer starts
			gettimeofday(&time_start1, NULL);


			cal_matrixmultion(temp, coef);


			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;

			//==== timer starts
			gettimeofday(&time_start1, NULL);


			float temp_value = temp.sum();


			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;

			//==== timer starts
			gettimeofday(&time_start1, NULL);


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


			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;

			//==== timer starts
			gettimeofday(&time_start1, NULL);



			//==##== collector ==##==
			//data.release();
			coef.release();
			temp.release();



			// //==== timer ends
			// gettimeofday(&time_end1, NULL);
			// time_diff1 = (double)(time_end1.tv_sec-time_start1.tv_sec) + (double)(time_end1.tv_usec-time_start1.tv_usec)/1000000;
			// printf("time used is %f seconds.\n", time_diff1);
			// cout << "###" << endl;


		}



		//==##== collector ==##==
		mean1.release();
		data.release();



		//==== timer ends
		gettimeofday(&time_end, NULL);
		time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
		printf("time used is %f seconds.\n", time_diff);
		cout << "@@@" << endl;


	}
	*/


	return;
}





// we have a specially designed function for Beta, as followed:
// NOTE: this is for uni-Gaussian prior with 0 mean, so mean 0 is fixed
void Gibbs_uniGaussian_Beta(
			float lambda_data,
			float lambda_prior,
			int d_factor,
			int d_snp,

			Matrix & fm,
			Matrix & U1_reshape,
			Matrix & X_reshape,
			Array & lambda_list,
			Matrix & mean_matrix)
{


	gpu_Gibbs_uniGaussian_Beta(lambda_data, lambda_prior, d_factor, d_snp, fm, U1_reshape, X_reshape, lambda_list, mean_matrix);



	/*
	for(int d=0; d<d_factor; d++)
	{
		float beta_last = 0;
		Array mean1 = extract_array_from_matrix(mean_matrix, d);

		Array data = extract_array_from_matrix(U1_reshape, d);

		for(int s=0; s<d_snp; s++)
		{
			//== prepare
			float x = fm.get_element(d, s);
			//Array data = extract_array_from_matrix(U1_reshape, d);

			if(s == 0)
			{
				beta_last = x;
			}
			else
			{
				float beta_new = fm.get_element(d, s-1);
				Array array = extract_array_from_matrix(X_reshape, s-1);
				cal_arrayaddon_multicoef(mean1, array, beta_new - beta_last);
				beta_last = x;
	
				//==##== collector ==##==
				array.release();
			}

			Array coef = extract_array_from_matrix(X_reshape, s);

			float lambda = lambda_list.get_element(s);
			float mu = 0;

			//== compute further
			Array temp = cal_arraysubtract(data, mean1);
			cal_arrayaddon_multicoef(temp, coef, x);
			cal_arraymultion(temp, coef);
			float temp_value = temp.sum();
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
			fm.set_element(d, s, draw);


			//==##== collector ==##==
			//data.release();
			coef.release();
			temp.release();
		}

		//==##== collector ==##==
		mean1.release();
		data.release();

	}
	*/



	return;
}





//// sampling 6 factors; might need some extra sub-routines
//==== manually set: order of tensor dimensions: K, I, J (tissue, individual, gene)
void sampler_factor(float lambda0, float mu1, float lambda1)
{
	cout << "sampling factors (Beta, U1, V1, T1, U2, V2, T2)..." << endl;



	// DEBUG
	struct timeval time_start;
	struct timeval time_end;
	double time_diff;






	//==== Beta
	cout << "sample Beta..." << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);



	//== U1_reshape
	Matrix U1_reshape = op_matrix_rotate(U1);



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);




	//== X_reshape
	Matrix X_reshape = op_matrix_rotate(X);



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);



	//== lambda_list
	Array lambda_list_beta;
	lambda_list_beta.init(S);
	for(int s=0; s<S; s++)
	{
		float value = lambda0 * X_reshape.sum_of_square(s) + lambda1;
		lambda_list_beta.set_element(s, value);
	}


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	// NOTE: for Beta (wide matrix), to avoid redundent computation; this could probably be used also for others
	Matrix mean_matrix = cal_matrixmul(Beta, X_reshape);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== sampling
	Gibbs_uniGaussian_Beta(lambda0, lambda1, D1, S, Beta, U1_reshape, X_reshape, lambda_list_beta, mean_matrix);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==##== collector ==##==
	U1_reshape.release();
	X_reshape.release();
	lambda_list_beta.release();
	mean_matrix.release();



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;








	//############################################
	//############################################
	//############################################
	//############################################







	//======== Y1 relevant
	//==== U1
	cout << "@@@@ sample U1..." << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== Y1_reshape
	Tensor Y1_reshape = op_tensor_reshape(Y1, 1, 0, 2);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== coef_tensor_reshape: (K, J, D)
	//Tensor coef_tensor_reshape = cal_tensorouter(T1, V1);
	//== coef_tensor: (D, K, J)
	Tensor coef_tensor = cal_tensorouter_reshape(T1, V1);



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== lambda_list
	Array lambda_list;
	lambda_list.init(D1);
	for(int d=0; d<D1; d++)
	{
		float temp = lambda0 * coef_tensor.sum_of_square(d) + lambda0;
		lambda_list.set_element(d, temp);
	}

	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== mu_matrix
	Matrix Beta_reshape = op_matrix_rotate(Beta);

	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	Matrix mu_matrix = cal_matrixmul(X, Beta_reshape);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== mean_tensor
	Tensor mean_tensor = cal_tensor_innerprod(T1, U1, V1);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	Tensor mean_tensor_reshape = op_tensor_reshape(mean_tensor, 1, 0, 2);
	// NOTE: we don't need mean_tensor any more
	mean_tensor.release();


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== sampling
	Gibbs_uniGaussian_fm_parallel(lambda0, lambda0, I, D1, U1, Y1_reshape, coef_tensor, lambda_list, mu_matrix, mean_tensor_reshape);

	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==##== collector ==##==
	Y1_reshape.release();
	coef_tensor.release();
	//coef_tensor_reshape.release();
	lambda_list.release();
	Beta_reshape.release();
	mu_matrix.release();
	//mean_tensor.release();
	mean_tensor_reshape.release();


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;








	//############################################
	//############################################
	//############################################
	//############################################







	//==== V1
	cout << "@@@@ sample V1..." << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== Y1_reshape
	Tensor Y1_reshape1 = op_tensor_reshape(Y1, 2, 0, 1);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== coef_tensor_reshape: (K, I, D)
	//Tensor coef_tensor_reshape1 = cal_tensorouter(T1, U1);
	//== coef_tensor: (D, K, I)
	Tensor coef_tensor1 = cal_tensorouter_reshape(T1, U1);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== lambda_list
	Array lambda_list1;
	lambda_list1.init(D1);
	for(int d=0; d<D1; d++)
	{
		float temp = lambda0 * coef_tensor1.sum_of_square(d) + lambda1;
		lambda_list1.set_element(d, temp);
	}


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);



	//== mu_matrix
	Matrix mu_matrix1;
	mu_matrix1.init(J, D1, mu1);		// to make function standard


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== mean_tensor
	Tensor mean_tensor1 = cal_tensor_innerprod(T1, U1, V1);



	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);


	Tensor mean_tensor_reshape1 = op_tensor_reshape(mean_tensor1, 2, 0, 1);
	// NOTE: we don't need mean_tensor1 any more
	mean_tensor1.release();


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== sampling
	Gibbs_uniGaussian_fm_parallel(lambda0, lambda1, J, D1, V1, Y1_reshape1, coef_tensor1, lambda_list1, mu_matrix1, mean_tensor_reshape1);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	//==##== collector ==##==
	Y1_reshape1.release();
	coef_tensor1.release();
	//coef_tensor_reshape1.release();
	lambda_list1.release();
	mu_matrix1.release();
	//mean_tensor1.release();
	mean_tensor_reshape1.release();


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;








	//############################################
	//############################################
	//############################################
	//############################################








	//==== T1
	cout << "@@@@ sample T1..." << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== Y1_reshape
	Tensor Y1_reshape2 = op_tensor_reshape(Y1, 0, 1, 2);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== coef_tensor_reshape: (I, J, D)
	//Tensor coef_tensor_reshape2 = cal_tensorouter(U1, V1);
	//== coef_tensor: (D, I, J)
	Tensor coef_tensor2 = cal_tensorouter_reshape(U1, V1);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);



	//== lambda_list
	Array lambda_list2;
	lambda_list2.init(D1);
	//
	for(int d=0; d<D1; d++)
	{
		float temp = lambda0 * coef_tensor2.sum_of_square(d) + lambda1;
		lambda_list2.set_element(d, temp);
	}
	// UPDATE the above with GPU (code below: it seems that the GPU doesn't help here)
	// float temp_array[D1];
	// gpu_tensor_sosq_at(temp_array, coef_tensor2);
	// for(int d=0; d<D1; d++)
	// {
	// 	float temp = lambda0 * temp_array[d] + lambda1;
	// 	lambda_list2.set_element(d, temp);
	// }
	


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== mu_matrix
	Matrix mu_matrix2;
	mu_matrix2.init(K, D1, mu1);		// to make function standard


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;


	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== mean_tensor
	Tensor mean_tensor2 = cal_tensor_innerprod(T1, U1, V1);			// GPU-accelerated


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;



	//==== timer starts
	gettimeofday(&time_start, NULL);


	Tensor mean_tensor_reshape2 = op_tensor_reshape(mean_tensor2, 0, 1, 2);
	// NOTE: we don't need mean_tensor2 any more
	mean_tensor2.release();


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);


	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda1, K, D1, T1, Y1_reshape2, coef_tensor2, lambda_list2, mu_matrix2, mean_tensor_reshape2);


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;




	//==== timer starts
	gettimeofday(&time_start, NULL);



	//==##== collector ==##==
	Y1_reshape2.release();
	coef_tensor2.release();
	//coef_tensor_reshape2.release();
	lambda_list2.release();
	mu_matrix2.release();
	//mean_tensor2.release();
	mean_tensor_reshape2.release();


	//==== timer ends
	gettimeofday(&time_end, NULL);
	time_diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("time used is %f seconds.\n", time_diff);
	cout << "####" << endl;









	// above: Beta, U1, V1, T1 (with timer)
	//############################################
	//############################################
	//############################################
	//############################################
	//############################################
	//############################################
	//############################################
	//############################################
	// above: U2, V2, T2 (without timer)








	//======== Y2 relevant
	//==== U2
	cout << "sample U2..." << endl;
	//== Y2_reshape
	Tensor Y2_reshape3 = op_tensor_reshape(Y2, 1, 0, 2);
	//== coef_tensor_reshape: (K, J, D)
	//Tensor coef_tensor_reshape3 = cal_tensorouter(T2, V2);
	//== coef_tensor: (D, K, J)
	Tensor coef_tensor3 = cal_tensorouter_reshape(T2, V2);
	//== lambda_list
	Array lambda_list3;
	lambda_list3.init(D2);
	for(int d=0; d<D2; d++)
	{
		float temp = lambda0 * coef_tensor3.sum_of_square(d) + lambda1;
		lambda_list3.set_element(d, temp);
	}
	//== mu_matrix
	Matrix mu_matrix3;
	mu_matrix3.init(I, D2, mu1);		// to make function standard
	//== mean_tensor
	Tensor mean_tensor3 = cal_tensor_innerprod(T2, U2, V2);
	Tensor mean_tensor_reshape3 = op_tensor_reshape(mean_tensor3, 1, 0, 2);
	// NOTE: we don't need mean_tensor3 any more
	mean_tensor3.release();
	//== sampling
	Gibbs_uniGaussian_fm_parallel(lambda0, lambda1, I, D2, U2, Y2_reshape3, coef_tensor3, lambda_list3, mu_matrix3, mean_tensor_reshape3);

	//==##== collector ==##==
	Y2_reshape3.release();
	coef_tensor3.release();
	//coef_tensor_reshape3.release();
	lambda_list3.release();
	mu_matrix3.release();
	//mean_tensor3.release();
	mean_tensor_reshape3.release();




	//==== V2
	cout << "sample V2..." << endl;
	//== Y2_reshape
	Tensor Y2_reshape4 = op_tensor_reshape(Y2, 2, 0, 1);
	//== coef_tensor_reshape: (K, I, D)
	//Tensor coef_tensor_reshape4 = cal_tensorouter(T2, U2);
	//== coef_tensor: (D, K, I)
	Tensor coef_tensor4 = cal_tensorouter_reshape(T2, U2);
	//== lambda_list
	Array lambda_list4;
	lambda_list4.init(D2);
	for(int d=0; d<D2; d++)
	{
		float temp = lambda0 * coef_tensor4.sum_of_square(d) + lambda1;
		lambda_list4.set_element(d, temp);
	}
	//== mu_matrix
	Matrix mu_matrix4;
	mu_matrix4.init(J, D2, mu1);		// to make function standard
	//== mean_tensor
	Tensor mean_tensor4 = cal_tensor_innerprod(T2, U2, V2);
	Tensor mean_tensor_reshape4 = op_tensor_reshape(mean_tensor4, 2, 0, 1);
	// NOTE: we don't need mean_tensor4 any more
	mean_tensor4.release();
	//== sampling
	Gibbs_uniGaussian_fm_parallel(lambda0, lambda1, J, D2, V2, Y2_reshape4, coef_tensor4, lambda_list4, mu_matrix4, mean_tensor_reshape4);

	//==##== collector ==##==
	Y2_reshape4.release();
	coef_tensor4.release();
	//coef_tensor_reshape4.release();
	lambda_list4.release();
	mu_matrix4.release();
	//mean_tensor4.release();
	mean_tensor_reshape4.release();




	//==== T2
	cout << "sample T2..." << endl;
	//== Y2_reshape
	Tensor Y2_reshape5 = op_tensor_reshape(Y2, 0, 1, 2);
	//== coef_tensor_reshape: (I, J, D)
	//Tensor coef_tensor_reshape5 = cal_tensorouter(U2, V2);
	//== coef_tensor: (D, I, J)
	Tensor coef_tensor5 = cal_tensorouter_reshape(U2, V2);
	//== lambda_list
	Array lambda_list5;
	lambda_list5.init(D2);
	for(int d=0; d<D2; d++)
	{
		float temp = lambda0 * coef_tensor5.sum_of_square(d) + lambda1;
		lambda_list5.set_element(d, temp);
	}
	//== mu_matrix
	Matrix mu_matrix5;
	mu_matrix5.init(K, D2, mu1);		// to make function standard
	//== mean_tensor
	Tensor mean_tensor5 = cal_tensor_innerprod(T2, U2, V2);
	Tensor mean_tensor_reshape5 = op_tensor_reshape(mean_tensor5, 0, 1, 2);
	// NOTE: we don't need mean_tensor5 any more
	mean_tensor5.release();
	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda1, K, D2, T2, Y2_reshape5, coef_tensor5, lambda_list5, mu_matrix5, mean_tensor_reshape5);

	//==##== collector ==##==
	Y2_reshape5.release();
	coef_tensor5.release();
	//coef_tensor_reshape5.release();
	lambda_list5.release();
	mu_matrix5.release();
	//mean_tensor5.release();
	mean_tensor_reshape5.release();





	return;
}




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============




//// sampling alpha, the precision
void sampler_precision(float alpha0, float beta0)
{
	//global Y, Y1, Y2, I, J, K, alpha, markerset, N_element
	cout << "sampling precision -- alpha ..." << endl;

	float alpha_new = alpha0 + 0.5 * N_element;

	Tensor temp0 = cal_tensorsubtract(Y, Y1);
	Tensor temp = cal_tensorsubtract(temp0, Y2);
	temp.power(2.0);
	cal_tensormultion(temp, markerset);
	float temp_value = temp.sum();
	float beta_new = beta0 + 0.5 * temp_value;

	float shape = alpha_new;
	float scale = 1.0 / beta_new;

	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	gamma_distribution<double> distribution(shape, scale);
    float draw = distribution(generator);

	alpha = draw;


	//==##== collector ==##==
	temp0.release();
	temp.release();


	return;
}




