/*

Thanks for coming! I know you will use some of my as utilities.

*/


#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "utility.h"
#include "utility_gpu.h"




using namespace std;




//==== indicating CPU or GPU (if appplicable)
// for:
//	void cal_arrayaddon_multicoef(Array, Array, float);
//	void cal_matrixaddon_multicoef(Matrix, Matrix, float);
//	Tensor cal_tensoradd_multicoef(float, Tensor, float, Tensor);
//
//	Array cal_arraysubtract(Array, Array);
//	Matrix cal_matrixsubtract(Matrix, Matrix);
//	Matrix cal_matrixsubtract(Matrix, float);
//	Tensor cal_tensorsubtract(Tensor, Tensor);
int indicator_GPU_saxpy = 0;								// NOTE: this seems not helping

// for:
//	void cal_arraymultion(Array, Array);
//	void cal_matrixmultion(Matrix, Matrix);
//	void cal_tensormultion(Tensor, Tensor);
//	Tensor cal_tensormultiply(Tensor, Tensor);
int indicator_GPU_multion = 0;								// NOTE: this seems not helping

// for:
//	Matrix cal_matrixmul(Matrix, Matrix);
int indicator_GPU_matrixmul = 1;							// NOTE: this helps for large dataset

// for:
//	Tensor cal_tensorouter(Matrix, Matrix);
int indicator_GPU_tensorouter = 0;							// NOTE: this helps !!!
															// NOTE: () is too large for GPU, so we won't use this with GPU

// for:
//	Tensor cal_tensor_innerprod(Matrix matrix1, Matrix matrix2, Matrix matrix3)
int indicator_GPU_tensor_innerprod = 1;						// NOTE: this helps !!!





//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============




// add the coef x Array array2 to the Array array1, in place for Array array1
void cal_arrayaddon_multicoef(Array array1, Array array2, float coef)
{
	int dimension = array1.get_dimension();

	if(!indicator_GPU_saxpy)						// CPU code
	{
		for(int i=0; i<dimension; i++)
		{
			float value = coef * array2.get_element(i);
			array1.addon(i, value);
		}
	}
	else 											// GPU code
	{
		float * pointer_array1 = array1.get_array();
		float * pointer_array2 = array2.get_array();
		gpu_saxpy_array(pointer_array1, pointer_array2, dimension, coef);
	}

	return;
}



// add the float coef x Matrix matrix2 to the Matrix matrix1, in place for Matrix matrix1
void cal_matrixaddon_multicoef(Matrix matrix1, Matrix matrix2, float coef)
{
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix1.get_dimension2();

	if(!indicator_GPU_saxpy)						// CPU code
	{
		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				matrix1.addon(i, j, coef * matrix2.get_element(i, j));
			}
		}
	}
	else 											// GPU code
	{
		float * pointer_matrix1 = matrix1.get_matrix();
		float * pointer_matrix2 = matrix2.get_matrix();
		gpu_saxpy_array(pointer_matrix1, pointer_matrix2, dimension1*dimension2, coef);
	}

	return;
}



// perform float coef1 * Tensor tensor1 + float coef2 * Tensor tensor2
Tensor cal_tensoradd_multicoef( float coef1, Tensor tensor1, float coef2, Tensor tensor2 )
{
	Tensor tensor;
	int dimension1 = tensor1.get_dimension1();
	int dimension2 = tensor1.get_dimension2();
	int dimension3 = tensor1.get_dimension3();

	tensor.init(dimension1, dimension2, dimension3);		// NOTE: this will automatically set all as 0

	if(!indicator_GPU_saxpy)						// CPU code
	{
		for(int k=0; k<dimension1; k++)
		{
			for(int i=0; i<dimension2; i++)
			{
				for(int j=0; j<dimension3; j++)
				{
					float value = coef1 * tensor1.get_element(k, i, j) + coef2 * tensor2.get_element(k, i, j);
					tensor.set_element(k, i, j, value);
				}
			}
		}
	}
	else 											// GPU code
	{
		float * pointer_tensor = tensor.get_tensor();		// NOTE: tensor has been set to 0, in init
		float * pointer_tensor1 = tensor1.get_tensor();
		float * pointer_tensor2 = tensor2.get_tensor();
		gpu_saxpy_array(pointer_tensor, pointer_tensor1, dimension1*dimension2*dimension3, coef1);
		gpu_saxpy_array(pointer_tensor, pointer_tensor2, dimension1*dimension2*dimension3, coef2);
	}

	return tensor;
}




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



// subtract the Array array2 from the Array array1, and return result in Array array
Array cal_arraysubtract(Array array1, Array array2)
{
	if(!indicator_GPU_saxpy)						// CPU code
	{
		Array array;
		int dimension = array1.get_dimension();
		array.init(dimension);

		for(int i=0; i<dimension; i++)
		{
			float value = array1.get_element(i) - array2.get_element(i);
			array.set_element(i, value);
		}

		return array;
	}
	else 											// GPU code
	{
		Array array;
		int dimension = array1.get_dimension();

		float * pointer_array1 = array1.get_array();
		array.init(dimension, pointer_array1);

		float * pointer_array = array.get_array();
		float * pointer_array2 = array2.get_array();
		gpu_saxpy_array(pointer_array, pointer_array2, dimension, -1);

		return array;
	}
}



// subtract the Matrix matrix2 from the Matrix matrix1, and return in Matrix matrix
Matrix cal_matrixsubtract(Matrix matrix1, Matrix matrix2)
{
	if(!indicator_GPU_saxpy)						// CPU code
	{
		Matrix matrix;
		int dimension1 = matrix1.get_dimension1();
		int dimension2 = matrix1.get_dimension2();
		matrix.init(dimension1, dimension2);

		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				float value = matrix1.get_element(i, j) - matrix2.get_element(i, j);
				matrix.set_element(i, j, value);
			}
		}

		return matrix;
	}
	else 											// GPU code
	{
		Matrix matrix;
		int dimension1 = matrix1.get_dimension1();
		int dimension2 = matrix1.get_dimension2();

		float * pointer_matrix1 = matrix1.get_matrix();
		matrix.init(dimension1, dimension2, pointer_matrix1);

		float * pointer_matrix = matrix.get_matrix();
		float * pointer_matrix2 = matrix2.get_matrix();
		gpu_saxpy_array(pointer_matrix, pointer_matrix2, dimension1*dimension2, -1);

		return matrix;
	}
}




// subtract the float value from the Matrix matrix1, and return in Matrix matrix
Matrix cal_matrixsubtract(Matrix matrix1, float value)
{
	if(!indicator_GPU_saxpy)						// CPU code
	{
		Matrix matrix;
		int dimension1 = matrix1.get_dimension1();
		int dimension2 = matrix1.get_dimension2();
		matrix.init(dimension1, dimension2);

		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				float temp = matrix1.get_element(i, j) - value;
				matrix.set_element(i, j, temp);
			}
		}

		return matrix;
	}
	else 											// GPU code
	{
		Matrix matrix;
		int dimension1 = matrix1.get_dimension1();
		int dimension2 = matrix1.get_dimension2();

		float * pointer_matrix1 = matrix1.get_matrix();
		matrix.init(dimension1, dimension2, pointer_matrix1);

		float * pointer_matrix = matrix.get_matrix();
		Matrix matrix2;
		matrix2.init(dimension1, dimension2, value);
		float * pointer_matrix2 = matrix2.get_matrix();
		gpu_saxpy_array(pointer_matrix, pointer_matrix2, dimension1*dimension2, -1);

		return matrix;
	}
}





// subtract Tensor tensor2 from Tensor tensor1, and return results in Tensor tensor
Tensor cal_tensorsubtract(Tensor tensor1, Tensor tensor2)
{
	if(!indicator_GPU_saxpy)						// CPU code
	{
		Tensor tensor;
		int dimension1 = tensor1.get_dimension1();
		int dimension2 = tensor1.get_dimension2();
		int dimension3 = tensor1.get_dimension3();
		tensor.init(dimension1, dimension2, dimension3);

		for(int k=0; k<dimension1; k++)
		{
			for(int i=0; i<dimension2; i++)
			{
				for(int j=0; j<dimension3; j++)
				{
					float value = tensor1.get_element(k, i, j) - tensor2.get_element(k, i, j);
					tensor.set_element(k, i, j, value);
				}
			}
		}

		return tensor;
	}
	else 											// GPU code
	{
		Tensor tensor;
		int dimension1 = tensor1.get_dimension1();
		int dimension2 = tensor1.get_dimension2();
		int dimension3 = tensor1.get_dimension3();

		float * pointer_tensor1 = tensor1.get_tensor();
		tensor.init(dimension1, dimension2, dimension3, pointer_tensor1);

		float * pointer_tensor = tensor.get_tensor();
		float * pointer_tensor2 = tensor2.get_tensor();
		gpu_saxpy_array(pointer_tensor, pointer_tensor2, dimension1*dimension2*dimension3, -1);

		return tensor;
	}
}




//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



//  element-wise product, in place on Array array1
void cal_arraymultion(Array array1, Array array2)
{
	int dimension = array1.get_dimension();

	if(!indicator_GPU_multion)						// CPU code
	{
		for(int i=0; i<dimension; i++)
		{
			array1.set_element( i, array1.get_element(i) * array2.get_element(i) );
		}
	}
	else 											// GPU code
	{
		float * result = array1.get_array();
		float * input = array2.get_array();

		gpu_array_multion(dimension, result, input);
	}

	return;
}



//  element-wise product, in place on Matrix matrix1
void cal_matrixmultion(Matrix matrix1, Matrix matrix2)
{
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix1.get_dimension2();
	int dimension = dimension1 * dimension2;

	if(!indicator_GPU_multion)						// CPU code
	{
		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				matrix1.set_element(i, j, matrix1.get_element(i, j) * matrix2.get_element(i, j));
			}
		}
	}
	else 											// GPU code
	{
		float * result = matrix1.get_matrix();
		float * input = matrix2.get_matrix();

		gpu_array_multion(dimension, result, input);
	}

	return;
}




//  element-wise product, in place on Tensor tensor1
void cal_tensormultion(Tensor tensor1, Tensor tensor2)
{
	int dimension1 = tensor1.get_dimension1();
	int dimension2 = tensor1.get_dimension2();
	int dimension3 = tensor1.get_dimension3();
	int dimension = dimension1 * dimension2 * dimension3;

	if(!indicator_GPU_multion)
	{
		for(int k=0; k<dimension1; k++)
		{
			for(int i=0; i<dimension2; i++)
			{
				for(int j=0; j<dimension3; j++)
				{
					tensor1.set_element(k, i, j, tensor1.get_element(k, i, j) * tensor2.get_element(k, i, j));
				}
			}
		}
	}
	else
	{
		float * result = tensor1.get_tensor();
		float * input = tensor2.get_tensor();

		gpu_array_multion(dimension, result, input);
	}

	return;
}




//  element-wise product, return Tensor tensor
Tensor cal_tensormultiply(Tensor tensor1, Tensor tensor2)
{
	if(!indicator_GPU_multion)
	{
		Tensor tensor;
		int dimension1 = tensor1.get_dimension1();
		int dimension2 = tensor1.get_dimension2();
		int dimension3 = tensor1.get_dimension3();
		tensor.init(dimension1, dimension2, dimension3);

		for(int k=0; k<dimension1; k++)
		{
			for(int i=0; i<dimension2; i++)
			{
				for(int j=0; j<dimension3; j++)
				{
					tensor.set_element(k, i, j, tensor1.get_element(k, i, j) * tensor2.get_element(k, i, j));
				}
			}
		}

		return tensor;
	}
	else
	{
		Tensor tensor;
		int dimension1 = tensor1.get_dimension1();
		int dimension2 = tensor1.get_dimension2();
		int dimension3 = tensor1.get_dimension3();
		int dimension = dimension1 * dimension2 * dimension3;

		float * pointer_tensor1 = tensor1.get_tensor();
		tensor.init(dimension1, dimension2, dimension3, pointer_tensor1);

		float * result = tensor.get_tensor();
		float * input = tensor2.get_tensor();

		gpu_array_multion(dimension, result, input);

		return tensor;
	}
}



//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



// do the usual matrix multiplication for Matrix matrix1 and Matrix matrix2
Matrix cal_matrixmul(Matrix matrix1, Matrix matrix2)
{
	Matrix result;
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix2.get_dimension2();
	int d_factor = matrix1.get_dimension2();
	result.init(dimension1, dimension2);


	if(!indicator_GPU_matrixmul)					// CPU compute
	{
		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				float value = 0;
				for(int d=0; d<d_factor; d++)
				{
					value += matrix1.get_element(i, d) * matrix2.get_element(d, j);
				}
				result.set_element(i, j, value);
			}
		}
	}
	else 											// GPU compute
	{
		sMatrixSize matrix_size;
		matrix_size.uiWA = d_factor;
		matrix_size.uiHA = dimension1;
		matrix_size.uiWB = dimension2;
		matrix_size.uiHB = d_factor;
		matrix_size.uiWC = matrix_size.uiWB;
		matrix_size.uiHC = matrix_size.uiHA;

		float * pointer_A = matrix1.get_matrix();
		float * pointer_B = matrix2.get_matrix();
		float * pointer_C = result.get_matrix();

		gpu_cal_matrixmul(pointer_A, pointer_B, pointer_C, matrix_size);
	}

	return result;
}



//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



// make Matrix matrix1 (d1, d2) and Matrix matrix2 (d3, d2) into a Tensor tensor (d1, d3, d2)
Tensor cal_tensorouter(Matrix matrix1, Matrix matrix2)
{
	Tensor tensor;
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix2.get_dimension1();
	int dimension3 = matrix1.get_dimension2();
	tensor.init(dimension1, dimension2, dimension3);

	if(!indicator_GPU_tensorouter)
	{
		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				for(int d=0; d<dimension3; d++)
				{
					tensor.set_element(i, j, d, matrix1.get_element(i, d) * matrix2.get_element(j, d));
				}
			}
		}
	}
	else
	{
		float * pointer_matrix1 = matrix1.get_matrix();
		float * pointer_matrix2 = matrix2.get_matrix();
		float * pointer_tensor = tensor.get_tensor();
		gpu_cal_tensorouter(pointer_matrix1, pointer_matrix2, pointer_tensor, dimension1, dimension2, dimension3);
	}

	return tensor;
}




// make Matrix matrix1 (d1, d2) and Matrix matrix2 (d3, d2) into a Tensor tensor (d3, d1, d2)
// since the dimension of the tensor for (400, 450, 21150) tensor, we won't use GPU for this routine
// NOTE: this routine currently needs 87.411926 secs for the (400, 450, 21150) matrix, which can be further improved
Tensor cal_tensorouter_reshape(Matrix matrix1, Matrix matrix2)
{
	Tensor tensor;
	int dimension1 = matrix1.get_dimension2();
	int dimension2 = matrix1.get_dimension1();
	int dimension3 = matrix2.get_dimension1();
	tensor.init(dimension1, dimension2, dimension3);

	for(int d=0; d<dimension1; d++)
	{
		for(int i=0; i<dimension2; i++)
		{
			for(int j=0; j<dimension3; j++)
			{
				tensor.set_element(d, i, j, matrix1.get_element(i, d) * matrix2.get_element(j, d));
			}
		}
	}

	return tensor;
}



//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



// inner product Matrix matrix1 (d1, d), Matrix matrix2 (d2, d) and Matrix matrix3 (d3, d), to a Tensor tensor (d1, d2, d3)
Tensor cal_tensor_innerprod(Matrix matrix1, Matrix matrix2, Matrix matrix3)
{
	Tensor tensor;
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix2.get_dimension1();
	int dimension3 = matrix3.get_dimension1();
	int d_factor = matrix1.get_dimension2();
	tensor.init(dimension1, dimension2, dimension3);

	if(!indicator_GPU_tensor_innerprod)
	{
		for(int k=0; k<dimension1; k++)
		{
			for(int i=0; i<dimension2; i++)
			{
				for(int j=0; j<dimension3; j++)
				{
					float value = 0;
					for(int d=0; d<d_factor; d++)
					{
						value += matrix1.get_element(k, d) * matrix2.get_element(i, d) * matrix3.get_element(j, d);
					}
					tensor.set_element(k, i, j, value);
				}
			}
		}
	}
	else
	{
		float * pointer_matrix1 = matrix1.get_matrix();
		float * pointer_matrix2 = matrix2.get_matrix();
		float * pointer_matrix3 = matrix3.get_matrix();
		float * pointer_tensor = tensor.get_tensor();

		gpu_cal_tensor_innerprod(pointer_matrix1, pointer_matrix2, pointer_matrix3, pointer_tensor, dimension1, dimension2, dimension3, d_factor);
	}

	return tensor;
}





// now I hard code this function to dot along (0, 2) axis (0 of array, and 2 of tensor)
Matrix cal_tensordot(Array array, Tensor tensor)
{
	Matrix matrix;
	int dimension1 = tensor.get_dimension1();
	int dimension2 = tensor.get_dimension2();
	int dimension3 = tensor.get_dimension3();
	matrix.init(dimension1, dimension2);

	for(int i=0; i<dimension1; i++)
	{
		for(int j=0; j<dimension2; j++)
		{
			float value = 0;
			for(int d=0; d<dimension3; d++)
			{
				value += array.get_element(d) * tensor.get_element(i, j, d);
			}
			matrix.set_element(i, j, value);
		}
	}

	return matrix;
}



//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



// rotate the input Matrix matrix
Matrix op_matrix_rotate(Matrix matrix)
{
	Matrix result;
	int dimension1 = matrix.get_dimension2();
	int dimension2 = matrix.get_dimension1();
	result.init(dimension1, dimension2);

	for(int i=0; i<dimension1; i++)
	{
		for(int j=0; j<dimension2; j++)
		{
			float value = matrix.get_element(j, i);
			result.set_element(i, j, value);
		}
	}
	return result;
}




// reshape a Tensor tensor with new shape as (d(pos1), d(pos2), d(pos3)), where d is the old shape
// NOTE: origianl order: (K, I, J)
Tensor op_tensor_reshape(Tensor tensor, int pos1, int pos2, int pos3)
{
	Tensor result;

	int dimension1 = tensor.get_dimension1();
	int dimension2 = tensor.get_dimension2();
	int dimension3 = tensor.get_dimension3();

	vector<int> vec;
	vec.push_back(dimension1);
	vec.push_back(dimension2);
	vec.push_back(dimension3);

	dimension1 = vec.at(pos1);
	dimension2 = vec.at(pos2);
	dimension3 = vec.at(pos3);

	result.init(dimension1, dimension2, dimension3);


	for(int i=0; i<dimension1; i++)
	{
		for(int j=0; j<dimension2; j++)
		{
			for(int d=0; d<dimension3; d++)
			{
				float value;

				// options
				if(pos1 == 0 && pos2 == 1 && pos3 == 2)
				{
					value = tensor.get_element(i, j, d);
				}
				if(pos1 == 0 && pos2 == 2 && pos3 == 1)
				{
					value = tensor.get_element(i, d, j);
				}
				if(pos1 == 1 && pos2 == 0 && pos3 == 2)
				{
					value = tensor.get_element(j, i, d);
				}
				if(pos1 == 1 && pos2 == 2 && pos3 == 0)
				{
					value = tensor.get_element(d, i, j);
				}
				if(pos1 == 2 && pos2 == 0 && pos3 == 1)
				{
					value = tensor.get_element(j, d, i);
				}
				if(pos1 == 2 && pos2 == 1 && pos3 == 0)
				{
					value = tensor.get_element(d, j, i);
				}

				result.set_element(i, j, d, value);
			}
		}
	}


	return result;
}



//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============
//=============/=============/=============/=============/=============/=============/=============/=============



// NOTE: the following two functions touch the internal contents of these Class, so need to be carefully used !!!
Array extract_array_from_matrix(Matrix matrix, int ind)
{
	Array array;
	int dimension2 = matrix.get_dimension2();

	int shift = ind * dimension2;
	float * pointer = matrix.get_matrix();
	array.init( dimension2, (pointer+shift) );

	return array;
}




Matrix extract_matrix_from_tensor(Tensor tensor, long int ind)
{
	Matrix matrix;
	long int dimension2 = tensor.get_dimension2();
	long int dimension3 = tensor.get_dimension3();

	long int shift = ind * dimension2 * dimension3;
	float * pointer = tensor.get_tensor();
	matrix.init( dimension2, dimension3, (pointer+shift) );

	return matrix;
}



