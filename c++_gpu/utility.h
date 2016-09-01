// utility.h
// function:

#ifndef UTILITY_H
#define UTILITY_H



#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       /* pow */
#include <string>
#include <string.h>
#include <vector>



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
		array = (float *)calloc( length, sizeof(float) );

		for(int i=0; i<length; i++)
		{
			array[i] = data[i];
		}
		return;
	}

	void print_shape()
	{
		cout << "this array has shape: " << dimension << endl;
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

	// given a filename, try to save this array into a file
	void save(char * filename)
	{
		FILE * file_out = fopen(filename, "w+");
		if(file_out == NULL)
		{
		    fputs("File error\n", stderr); exit(1);
		}

		for(int i=0; i<dimension; i++)
		{
			float value = array[i];
			char buf[1024];
			sprintf(buf, "%f\t", value);
			fwrite(buf, sizeof(char), strlen(buf), file_out);
		}
		fwrite("\n", sizeof(char), 1, file_out);
		fclose(file_out);

		return;
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
		matrix = (float *)calloc( length1 * length2, sizeof(float) );

		for(int i=0; i<length1*length2; i++)
		{
			matrix[i] = data[i];
		}
		return;
	}

	// used for loading data, since we might use other containers to load data first of all
	void init(vector<vector<float>> & container)
	{
		dimension1 = container.size();
		dimension2 = (container.at(0)).size();
		matrix = (float *)calloc( dimension1 * dimension2, sizeof(float) );

		int count = 0;
		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				float value = (container.at(i)).at(j);
				matrix[count] = value;
				count += 1;
			}
		}

		return;
	}

	void print_shape()
	{
		cout << "this matrix has shape: " << dimension1 << ", " << dimension2 << endl;
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
	/* following couldn't work
	Array get_array(int ind)
	{
		Array array;
		int shift = ind * dimension2;
		array.init( dimension2, (matrix+shift) );
		return array;
	}
	*/

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

	// given a filename, try to save this matrix into a file
	void save(char * filename)
	{
		FILE * file_out = fopen(filename, "w+");
		if(file_out == NULL)
		{
		    fputs("File error\n", stderr); exit(1);
		}

		for(int i=0; i<dimension1; i++)
		{
			int start = i * dimension2;
			for(int j=0; j<dimension2; j++)
			{
				int index = start + j;
				float value = matrix[index];
				char buf[1024];
				sprintf(buf, "%f\t", value);
				fwrite(buf, sizeof(char), strlen(buf), file_out);
			}
			fwrite("\n", sizeof(char), 1, file_out);
		}
		fclose(file_out);
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

	void init(int length1, int length2, int length3, float value)
	{
		dimension1 = length1;
		dimension2 = length2;
		dimension3 = length3;
		tensor = (float *)calloc( length1 * length2 * length3, sizeof(float) );
		for(int i=0; i<dimension1*dimension2*dimension3; i++)
		{
			tensor[i] = value;
		}

		return;
	}

	// used for loading data, since we might use other containers to load data first of all
	void init(vector<vector<vector<float>>> & container)
	{
		dimension1 = container.size();
		dimension2 = (container.at(0)).size();
		dimension3 = ((container.at(0)).at(0)).size();
		tensor = (float *)calloc( dimension1 * dimension2 * dimension3, sizeof(float) );

		int count = 0;
		for(int i=0; i<dimension1; i++)
		{
			for(int j=0; j<dimension2; j++)
			{
				for(int d=0; d<dimension3; d++)
				{
					float value = ((container.at(i)).at(j)).at(d);
					tensor[count] = value;
					count += 1;
				}
			}
		}

		return;
	}

	void print_shape()
	{
		cout << "this tensor has shape: " << dimension1 << ", " << dimension2 << ", " << dimension3 << endl;
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

	/* following couldn't work
	Matrix get_matrix(int ind)
	{
		Matrix matrix;
		int shift = ind * dimension2 * dimension3;
		matrix.init( dimension2, dimension3, (tensor+shift) );
		return matrix;
	}
	*/

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

	// given a filename, try to save this tensor into a file
	void save(char * filename)
	{
		FILE * file_out = fopen(filename, "w+");
		if(file_out == NULL)
		{
		    fputs("File error\n", stderr); exit(1);
		}

		char buf[1024];
		sprintf(buf, "%d\t", dimension1);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
		sprintf(buf, "%d\t", dimension2);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
		sprintf(buf, "%d\t", dimension3);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
		fwrite("\n", sizeof(char), 1, file_out);

		for(int k=0; k<dimension1; k++)
		{
			for(int i=0; i<dimension2; i++)
			{
				int start = k * dimension2 * dimension3 + i * dimension3;
				for(int j=0; j<dimension3; j++)
				{
					int index = start + j;
					float value = tensor[index];
					char buf[1024];
					sprintf(buf, "%f\t", value);
					fwrite(buf, sizeof(char), strlen(buf), file_out);
				}
				fwrite("\n", sizeof(char), 1, file_out);
			}
		}
		fclose(file_out);
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

Array cal_arraysubtract(Array, Array);
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
Matrix cal_tensordot(Array, Tensor);

Matrix op_matrix_rotate(Matrix);
Tensor op_tensor_reshape(Tensor, int, int, int);

Array extract_array_from_matrix(Matrix, int);
Matrix extract_matrix_from_tensor(Tensor, int);




#endif

// end of utility.h
