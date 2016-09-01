/*

Aha, here comes the GIbbs sampling program


*/


// system
#include <iostream>
#include <math.h>       /* sqrt */
#include <random>
#include <chrono>		/* sys time */

// private
#include "global.h"
#include "utility.h"




using namespace std;





//// sampling two sub-tensors, Y1 and Y2
void sampler_subT(float lambda)
{
	cout << "sampling sub-tensors..." << endl;
	//global Y, Y1, Y2, U1, V1, T1, U2, V2, T2, I, J, K, alpha, markerset


	//==== sample Y1
	cout << "sample Y1..." << endl;
	//== pre-cal
	Tensor mean1 = cal_tensor_innerprod(T1, U1, V1);
	Tensor mean2 = cal_tensorsubtract(Y, Y2);
	//== combine two Gaussian's
	float lambda1 = lambda;
	float lambda2 = alpha;
	float lambda_new = lambda1 + lambda2;
	Tensor mean_new = cal_tensoradd_multicoef( (lambda1 / lambda_new), mean1, (lambda2 / lambda_new), mean2 );
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






// the following samples the uniGaussian factor matrix (uni-variance for both data and prior)
void Gibbs_uniGaussian_fm(
			float lambda_data,
			float lambda_prior,
			int d_feature,
			int d_factor,
			Matrix & fm,
			Tensor & tensor_reshape,
			Tensor & coef_tensor,
			Tensor & coef_tensor_reshape,
			Array & lambda_list,
			Matrix & mu_matrix,
			Tensor & mean_tensor)
{
	float entry_last = 0;
	// need: mean_tensor
	Matrix mean1;

	for(int i=0; i<d_feature; i++)
	{
		for(int d=0; d<d_factor; d++)
		{
			//== prepare
			float x = fm.get_element(i, d);
			Matrix data = extract_matrix_from_tensor(tensor_reshape, i);

			//Array array = extract_array_from_matrix(fm, i);
			//Matrix mean1 = cal_tensordot(array, coef_tensor_reshape);
			if(d == 0)
			{
				mean1 = extract_matrix_from_tensor(mean_tensor, i);
				entry_last = x;
			}
			else
			{
				float entry_new = fm.get_element(i, d-1);
				Matrix matrix = extract_matrix_from_tensor(coef_tensor, d-1);
				cal_matrixaddon_multicoef(mean1, matrix, entry_new - entry_last);
				entry_last = x;

				//==##== collector ==##==
				matrix.release();
			}

			Matrix coef = extract_matrix_from_tensor(coef_tensor, d);

			float lambda = lambda_list.get_element(d);
			float mu = mu_matrix.get_element(i, d);

			//== compute further
			Matrix temp = cal_matrixsubtract(data, mean1);
			cal_matrixaddon_multicoef(temp, coef, x);
			cal_matrixmultion(temp, coef);
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
			fm.set_element(i, d, draw);


			//==##== collector ==##==
			data.release();
			coef.release();
			temp.release();
		}
	}


	//==##== collector ==##==
	mean1.release();


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
			Array & lambda_list)
{
	// NOTE: for Beta (wide matrix), to avoid redundent computation; this could probably be used also for others
	float beta_last = 0;
	Matrix mean_matrix = cal_matrixmul(Beta, X_reshape);
	Array mean1;

	for(int d=0; d<d_factor; d++)
	{
		for(int s=0; s<d_snp; s++)
		{
			//== prepare
			float x = fm.get_element(d, s);
			Array data = extract_array_from_matrix(U1_reshape, d);

			if(s == 0)
			{
				mean1 = extract_array_from_matrix(mean_matrix, d);
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
			data.release();
			coef.release();
			temp.release();
		}
	}

	//==##== collector ==##==
	mean_matrix.release();
	mean1.release();


	return;
}





//// sampling 6 factors; might need some extra sub-routines
//==== manually set: order of tensor dimensions: K, I, J (tissue, individual, gene)
void sampler_factor(float lambda0, float mu1, float lambda1)
{
	cout << "sampling factors (Beta, U1, V1, T1, U2, V2, T2)..." << endl;

	//==== Beta
	cout << "sample Beta..." << endl;
	//== U1_reshape
	Matrix U1_reshape = op_matrix_rotate(U1);
	//== X_reshape
	Matrix X_reshape = op_matrix_rotate(X);
	//== lambda_list
	Array lambda_list;
	lambda_list.init(S);
	for(int s=0; s<S; s++)
	{
		float value = lambda0 * X_reshape.sum_of_square(s) + lambda1;
		lambda_list.set_element(s, value);
	}
	//== sampling
	Gibbs_uniGaussian_Beta(lambda0, lambda1, D1, S, Beta, U1_reshape, X_reshape, lambda_list);

	//==##== collector ==##==
	U1_reshape.release();
	X_reshape.release();
	lambda_list.release();




	//======== Y1 relevant
	//==== U1
	cout << "sample U1..." << endl;
	//== Y1_reshape
	Tensor Y1_reshape = op_tensor_reshape(Y1, 1, 0, 2);
	//== coef_tensor_reshape: (K, J, D)
	Tensor coef_tensor_reshape = cal_tensorouter(T1, V1);
	//== coef_tensor: (D, K, J)
	Tensor coef_tensor = op_tensor_reshape(coef_tensor_reshape, 2, 0, 1);
	//== lambda_list
	//Array lambda_list;
	lambda_list.init(D1);
	for(int d=0; d<D1; d++)
	{
		float temp = lambda0 * coef_tensor.sum_of_square(d) + lambda0;
		lambda_list.set_element(d, temp);
	}
	//== mu_matrix
	Matrix Beta_reshape = op_matrix_rotate(Beta);
	Matrix mu_matrix = cal_matrixmul(X, Beta_reshape);
	//== mean_tensor
	Tensor mean_tensor = cal_tensor_innerprod(T1, U1, V1);
	Tensor mean_tensor_reshape = op_tensor_reshape(mean_tensor, 1, 0, 2);
	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda0, I, D1, U1, Y1_reshape, coef_tensor, coef_tensor_reshape, lambda_list, mu_matrix, mean_tensor_reshape);

	//==##== collector ==##==
	Y1_reshape.release();
	coef_tensor.release();
	coef_tensor_reshape.release();
	lambda_list.release();
	Beta_reshape.release();
	mu_matrix.release();
	mean_tensor.release();
	mean_tensor_reshape.release();




	//==== V1
	cout << "sample V1..." << endl;
	//== Y1_reshape
	//Tensor Y1_reshape = op_tensor_reshape(Y1, 2, 0, 1);
	Y1_reshape = op_tensor_reshape(Y1, 2, 0, 1);
	//== coef_tensor_reshape: (K, I, D)
	coef_tensor_reshape = cal_tensorouter(T1, U1);
	//== coef_tensor: (D, K, I)
	coef_tensor = op_tensor_reshape(coef_tensor_reshape, 2, 0, 1);
	//== lambda_list
	//Array lambda_list;
	lambda_list.init(D1);
	for(int d=0; d<D1; d++)
	{
		float temp = lambda0 * coef_tensor.sum_of_square(d) + lambda1;
		lambda_list.set_element(d, temp);
	}
	//== mu_matrix
	//Matrix mu_matrix;
	mu_matrix.init(J, D1, mu1);		// to make function standard
	//== mean_tensor
	mean_tensor = cal_tensor_innerprod(T1, U1, V1);
	mean_tensor_reshape = op_tensor_reshape(mean_tensor, 2, 0, 1);
	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda1, J, D1, V1, Y1_reshape, coef_tensor, coef_tensor_reshape, lambda_list, mu_matrix, mean_tensor_reshape);

	//==##== collector ==##==
	Y1_reshape.release();
	coef_tensor.release();
	coef_tensor_reshape.release();
	lambda_list.release();
	mu_matrix.release();
	mean_tensor.release();
	mean_tensor_reshape.release();




	//==== T1
	cout << "sample T1..." << endl;
	//== Y1_reshape
	Y1_reshape = op_tensor_reshape(Y1, 0, 1, 2);
	//== coef_tensor_reshape: (I, J, D)
	coef_tensor_reshape = cal_tensorouter(U1, V1);
	//== coef_tensor: (D, I, J)
	coef_tensor = op_tensor_reshape(coef_tensor_reshape, 2, 0, 1);
	//== lambda_list
	//Array lambda_list;
	lambda_list.init(D1);
	for(int d=0; d<D1; d++)
	{
		float temp = lambda0 * coef_tensor.sum_of_square(d) + lambda1;
		lambda_list.set_element(d, temp);
	}
	//== mu_matrix
	//Matrix mu_matrix;
	mu_matrix.init(K, D1, mu1);		// to make function standard
	//== mean_tensor
	mean_tensor = cal_tensor_innerprod(T1, U1, V1);
	mean_tensor_reshape = op_tensor_reshape(mean_tensor, 0, 1, 2);
	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda1, K, D1, T1, Y1_reshape, coef_tensor, coef_tensor_reshape, lambda_list, mu_matrix, mean_tensor_reshape);

	//==##== collector ==##==
	Y1_reshape.release();
	coef_tensor.release();
	coef_tensor_reshape.release();
	lambda_list.release();
	mu_matrix.release();
	mean_tensor.release();
	mean_tensor_reshape.release();





	//======== Y2 relevant
	//==== U2
	cout << "sample U2..." << endl;
	//== Y2_reshape
	//Tensor Y2_reshape = op_tensor_reshape(Y2, 1, 0, 2);
	Tensor Y2_reshape = op_tensor_reshape(Y2, 1, 0, 2);
	//== coef_tensor_reshape: (K, J, D)
	coef_tensor_reshape = cal_tensorouter(T2, V2);
	//== coef_tensor: (D, K, J)
	coef_tensor = op_tensor_reshape(coef_tensor_reshape, 2, 0, 1);
	//== lambda_list
	//Array lambda_list;
	lambda_list.init(D2);
	for(int d=0; d<D2; d++)
	{
		float temp = lambda0 * coef_tensor.sum_of_square(d) + lambda1;
		lambda_list.set_element(d, temp);
	}
	//== mu_matrix
	//Matrix mu_matrix;
	mu_matrix.init(I, D2, mu1);		// to make function standard
	//== mean_tensor
	mean_tensor = cal_tensor_innerprod(T2, U2, V2);
	mean_tensor_reshape = op_tensor_reshape(mean_tensor, 1, 0, 2);
	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda1, I, D2, U2, Y2_reshape, coef_tensor, coef_tensor_reshape, lambda_list, mu_matrix, mean_tensor_reshape);

	//==##== collector ==##==
	Y2_reshape.release();
	coef_tensor.release();
	coef_tensor_reshape.release();
	lambda_list.release();
	mu_matrix.release();
	mean_tensor.release();
	mean_tensor_reshape.release();




	//==== V2
	cout << "sample V2..." << endl;
	//== Y2_reshape
	//Tensor Y2_reshape = op_tensor_reshape(Y2, 2, 0, 1);
	Y2_reshape = op_tensor_reshape(Y2, 2, 0, 1);
	//== coef_tensor_reshape: (K, I, D)
	coef_tensor_reshape = cal_tensorouter(T2, U2);
	//== coef_tensor: (D, K, I)
	coef_tensor = op_tensor_reshape(coef_tensor_reshape, 2, 0, 1);
	//== lambda_list
	//Array lambda_list;
	lambda_list.init(D2);
	for(int d=0; d<D2; d++)
	{
		float temp = lambda0 * coef_tensor.sum_of_square(d) + lambda1;
		lambda_list.set_element(d, temp);
	}
	//== mu_matrix
	//Matrix mu_matrix;
	mu_matrix.init(J, D2, mu1);		// to make function standard
	//== mean_tensor
	mean_tensor = cal_tensor_innerprod(T2, U2, V2);
	mean_tensor_reshape = op_tensor_reshape(mean_tensor, 2, 0, 1);
	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda1, J, D2, V2, Y2_reshape, coef_tensor, coef_tensor_reshape, lambda_list, mu_matrix, mean_tensor_reshape);

	//==##== collector ==##==
	Y2_reshape.release();
	coef_tensor.release();
	coef_tensor_reshape.release();
	lambda_list.release();
	mu_matrix.release();
	mean_tensor.release();
	mean_tensor_reshape.release();




	//==== T2
	cout << "sample T2..." << endl;
	//== Y2_reshape
	Y2_reshape = op_tensor_reshape(Y2, 0, 1, 2);
	//== coef_tensor_reshape: (I, J, D)
	coef_tensor_reshape = cal_tensorouter(U2, V2);
	//== coef_tensor: (D, I, J)
	coef_tensor = op_tensor_reshape(coef_tensor_reshape, 2, 0, 1);
	//== lambda_list
	//Array lambda_list;
	lambda_list.init(D2);
	for(int d=0; d<D2; d++)
	{
		float temp = lambda0 * coef_tensor.sum_of_square(d) + lambda1;
		lambda_list.set_element(d, temp);
	}
	//== mu_matrix
	//Matrix mu_matrix;
	mu_matrix.init(K, D2, mu1);		// to make function standard
	//== mean_tensor
	mean_tensor = cal_tensor_innerprod(T2, U2, V2);
	mean_tensor_reshape = op_tensor_reshape(mean_tensor, 0, 1, 2);
	//== sampling
	Gibbs_uniGaussian_fm(lambda0, lambda1, K, D2, T2, Y2_reshape, coef_tensor, coef_tensor_reshape, lambda_list, mu_matrix, mean_tensor_reshape);

	//==##== collector ==##==
	Y2_reshape.release();
	coef_tensor.release();
	coef_tensor_reshape.release();
	lambda_list.release();
	mu_matrix.release();
	mean_tensor.release();
	mean_tensor_reshape.release();




	return;
}





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




