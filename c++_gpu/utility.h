// utility.h
// function:

#ifndef UTILITY_H
#define UTILITY_H



#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       /* pow */
#include "global.h"



using namespace std;



//================
class Array
{
	int dimension;
	float * array;

public:
	void init(int length)
	{
		dimension = length;
		array = (float *)calloc( length, sizeof(float) );
		return;
	}

	void init(int length, float * data)
	{
		dimension = length;
		for(int i=0; i<length; i++)
		{
			array[i] = data[i];
		}
		return;
	}

	int get_dimension()
	{
		return dimension;
	}

	// NOTE: use the following function carefully -- do not modify elements from outside
	float * get_array()
	{
		return array;
	}

	float get_element(int ind)
	{
		return array[ind];
	}

	void set_element(int ind, float value)
	{
		array[ind] = value;
		return;
	}

	void addon(int ind, float value)
	{
		array[ind] += value;
		return;
	}

	float sum()
	{
		float result = 0;
		for(int i=0; i<dimension; i++)
		{
			result += array[i];
		}
		return result;
	}

	// SOS for all elements
	float sum_of_square()
	{
		float result = 0;
		for(int i=0; i<dimension; i++)
		{
			result += pow(array[i], 2.0);
		}
		return result;
	}

	// delete the object
	void release()
	{
		free(array);
		return;
	}

};



//================
class Matrix
{
	int dimension1;
	int dimension2;
	float * matrix;

public:
	void init(int length1, int length2, float value)
	{
		dimension1 = length1;
		dimension2 = length2;
		matrix = (float *)calloc( length1 * length2, sizeof(float) );
		for(int i=0; i<length1*length2; i++)
		{
			matrix[i] = value;
		}
		return;
	}

	void init(int length1, int length2)
	{
		dimension1 = length1;
		dimension2 = length2;
		matrix = (float *)calloc( length1 * length2, sizeof(float) );
		return;
	}

	void init(int length1, int length2, float * data)
	{
		dimension1 = length1;
		dimension2 = length2;

		for(int i=0; i<length1*length2; i++)
		{
			matrix[i] = data[i];
		}
		return;
	}

	int get_dimension1()
	{
		return dimension1;
	}

	int get_dimension2()
	{
		return dimension2;
	}

	// NOTE: use the following function carefully -- do not modify elements from outside
	float * get_matrix()
	{
		return matrix;
	}

	float get_element(int ind1, int ind2)
	{
		int index = ind1 * dimension2 + ind2;
		return matrix[index];
	}

	void set_element(int ind1, int ind2, float value)
	{
		int index = ind1 * dimension2 + ind2;
		matrix[index] = value;
		return;
	}

	// add on a value to the matrix entry, in place
	void addon(int ind1, int ind2, float value)
	{
		int index = ind1 * dimension2 + ind2;
		matrix[index] += value;
		return;
	}

	// pick up one array from Matrix and return Array
	Array get_array(int ind)
	{
		Array array;
		int shift = ind * dimension2;
		array.init( dimension2, (matrix+shift) );
		return array;
	}

	// sum all elements
	float sum()
	{
		float result = 0;
		for(int i=0; i<dimension1*dimension2; i++)
		{
			result += matrix[i];
		}
		return result;
	}

	// SOS for all elements
	float sum_of_square()
	{
		float result = 0;
		for(int i=0; i<dimension1*dimension2; i++)
		{
			result += pow(matrix[i], 2.0);
		}
		return result;
	}

	// SOS for on array
	float sum_of_square(int ind)
	{
		float result = 0;
		int start = ind * dimension2;
		for(int i=0; i<dimension2; i++)
		{
			result += pow(matrix[start+i], 2.0);
		}
		return result;
	}

	// raise all elements to the power of float exp
	void power(float exp)
	{
		for(int i=0; i<dimension1*dimension2; i++)
		{
			matrix[i] = pow(matrix[i], exp);
		}
		return;
	}

	// times all elements with coef
	void multiply(float coef)
	{
		for(int i=0; i<dimension1*dimension2; i++)
		{
			matrix[i] = matrix[i] * coef;
		}
		return;
	}


	// delete object
	void release()
	{
		free(matrix);
		return;
	}

};



//================
class Tensor
{
	int dimension1;
	int dimension2;
	int dimension3;
	float * tensor;

public:
	void init(int length1, int length2, int length3)
	{
		dimension1 = length1;
		dimension2 = length2;
		dimension3 = length3;
		tensor = (float *)calloc( length1 * length2 * length3, sizeof(float) );
		return;
	}

	int get_dimension1()
	{
		return dimension1;
	}

	int get_dimension2()
	{
		return dimension2;
	}

	int get_dimension3()
	{
		return dimension3;
	}

	// NOTE: use the following function carefully -- do not modify elements from outside
	float * get_tensor()
	{
		return tensor;
	}

	float get_element(int ind1, int ind2, int ind3)
	{
		int index = ind1 * dimension2 * dimension3 + ind2 * dimension3 + ind3;
		return tensor[index];
	}

	void set_element(int ind1, int ind2, int ind3, float value)
	{
		int index = ind1 * dimension2 * dimension3 + ind2 * dimension3 + ind3;
		tensor[index] = value;
		return;
	}

	Matrix get_matrix(int ind)
	{
		Matrix matrix;
		int shift = ind * dimension2 * dimension3;
		matrix.init( dimension2, dimension3, (tensor+shift) );
		return matrix;
	}

	float sum()
	{
		float result = 0;
		for(int i=0; i<dimension1*dimension2*dimension3; i++)
		{
			result += tensor[i];
		}
		return result;
	}

	// SOS for one layer
	float sum_of_square(int ind)
	{
		float result = 0;
		int start = ind * dimension2 * dimension3;
		for(int i=0; i<dimension2*dimension3; i++)
		{
			result += pow(tensor[start + i], 2.0);
		}
		return result;
	}

	// raise all elements to the power of float exp
	void power(float exp)
	{
		for(int i=0; i<dimension1*dimension2*dimension3; i++)
		{
			tensor[i] = pow(tensor[i], exp);
		}
		return;
	}


	// delete object
	void release()
	{
		free(tensor);
		return;
	}


};





void cal_arrayaddon_multicoef(Array, Array, float);
void cal_matrixaddon_multicoef(Matrix, Matrix, float);
Tensor cal_tensoradd_multicoef(float, Tensor, float, Tensor);

Matrix cal_matrixsubtract(Matrix, Matrix);
Matrix cal_matrixsubtract(Matrix, float);
Tensor cal_tensorsubtract(Tensor, Tensor);

void cal_arraymultion(Array, Array);
void cal_matrixmultion(Matrix, Matrix);
void cal_tensormultion(Tensor, Tensor);
Tensor cal_tensormultiply(Tensor, Tensor);

Matrix cal_matrixmul(Matrix, Matrix);

Tensor cal_tensorouter(Matrix, Matrix);
Tensor cal_tensor_innerprod(Matrix, Matrix, Matrix);
Matrix cal_tensordot(Array, Tensor, int, int);

Matrix op_matrix_rotate(Matrix);
Tensor op_tensor_reshape(Tensor, int, int, int);






#endif

// end of utility.h
