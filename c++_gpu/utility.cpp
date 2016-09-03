/*

Thanks for coming! I know you will use some of my as utilities.

*/



#include <iostream>
#include <vector>
#include "utility.h"
#include "utility_gpu.cuh"




using namespace std;




//==== indicating CPU or GPU (if appplicable)
int indicator_GPU = 1;






// add the coef x Array array2 to the Array array1
void cal_arrayaddon_multicoef(Array array1, Array array2, float coef)
{
	int dimension = array1.get_dimension();

	for(int i=0; i<dimension; i++)
	{
		float value = coef * array2.get_element(i);
		array1.addon(i, value);
	}

	return;
}



// add the float coef x Matrix matrix2 to the Matrix matrix1, in place for Matrix matrix1
void cal_matrixaddon_multicoef(Matrix matrix1, Matrix matrix2, float coef)
{
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix1.get_dimension2();

	for(int i=0; i<dimension1; i++)
	{
		for(int j=0; j<dimension2; j++)
		{
			matrix1.addon(i, j, coef * matrix2.get_element(i, j));
		}
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

	tensor.init(dimension1, dimension2, dimension3);

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

	return tensor;
}



// subtract the Array array2 from the Array array1
Array cal_arraysubtract(Array array1, Array array2)
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



// subtract the Matrix matrix2 from the Matrix matrix1
Matrix cal_matrixsubtract(Matrix matrix1, Matrix matrix2)
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


// subtract the float value from the Matrix matrix1
Matrix cal_matrixsubtract(Matrix matrix1, float value)
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



// subtract Tensor tensor2 from Tensor tensor1, and return results in Tensor tensor
Tensor cal_tensorsubtract(Tensor tensor1, Tensor tensor2)
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



//  element-wise product, in place on Array array1
void cal_arraymultion(Array array1, Array array2)
{
	int dimension = array1.get_dimension();

	for(int i=0; i<dimension; i++)
	{
		array1.set_element( i, array1.get_element(i) * array2.get_element(i) );
	}

	return;
}



//  element-wise product, in place on Matrix matrix1
void cal_matrixmultion(Matrix matrix1, Matrix matrix2)
{
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix1.get_dimension2();

	for(int i=0; i<dimension1; i++)
	{
		for(int j=0; j<dimension2; j++)
		{
			matrix1.set_element(i, j, matrix1.get_element(i, j) * matrix2.get_element(i, j));
		}
	}

	return;
}




//  element-wise product, in place on Tensor tensor1
void cal_tensormultion(Tensor tensor1, Tensor tensor2)
{
	int dimension1 = tensor1.get_dimension1();
	int dimension2 = tensor1.get_dimension2();
	int dimension3 = tensor1.get_dimension3();

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

	return;
}



//  element-wise product, return Tensor tensor
Tensor cal_tensormultiply(Tensor tensor1, Tensor tensor2)
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




// do the usual matrix multiplication for Matrix matrix1 and Matrix matrix2
Matrix cal_matrixmul(Matrix matrix1, Matrix matrix2)
{
	Matrix result;
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix2.get_dimension2();
	int d_factor = matrix1.get_dimension2();
	result.init(dimension1, dimension2);


	if(!indicator_GPU)			// CPU compute
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
	else 						// GPU compute
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



// make Matrix matrix1 (d1, d2) and Matrix matrix2 (d3, d2) into a Tensor result (d1, d3, d2)
Tensor cal_tensorouter(Matrix matrix1, Matrix matrix2)
{
	Tensor result;
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix2.get_dimension1();
	int dimension3 = matrix1.get_dimension2();
	result.init(dimension1, dimension2, dimension3);

	for(int i=0; i<dimension1; i++)
	{
		for(int j=0; j<dimension2; j++)
		{
			for(int d=0; d<dimension3; d++)
			{
				result.set_element(i, j, d, matrix1.get_element(i, d) * matrix2.get_element(j ,d));
			}
		}
	}
	return result;
}



// inner product Matrix matrix1 (d1, d), Matrix matrix2 (d2, d) and Matrix matrix3 (d3, d), to a Tensor tensor (d1, d2, d3)
Tensor cal_tensor_innerprod(Matrix matrix1, Matrix matrix2, Matrix matrix3)
{
	Tensor tensor;
	int dimension1 = matrix1.get_dimension1();
	int dimension2 = matrix2.get_dimension1();
	int dimension3 = matrix3.get_dimension1();
	int d_factor = matrix1.get_dimension2();
	tensor.init(dimension1, dimension2, dimension3);

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




Matrix extract_matrix_from_tensor(Tensor tensor, int ind)
{
	Matrix matrix;
	int dimension2 = tensor.get_dimension2();
	int dimension3 = tensor.get_dimension3();

	int shift = ind * dimension2 * dimension3;
	float * pointer = tensor.get_tensor();
	matrix.init( dimension2, dimension3, (pointer+shift) );

	return matrix;
}



