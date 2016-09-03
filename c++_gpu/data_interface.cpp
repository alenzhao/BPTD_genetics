/*

Thanks for stopping by!

The format of the input/output data:
1. Everything in Matrix format should be in matrix format
2. The tensor has one line meta information, the shape of the tensor, and then the tensor is spread as a matrix (collapsed the first dimension)
3. I will also re-prepare the incomplete tensor such that there are only numbers: integer-indexing tissues as usual; inside each tissue, integer indexing individual positions to have this expression sample (one extra integer at the head of each line)

*/



#include <iostream>
#include <string>		/* stof, stod */
#include <stdlib.h>     /* atoi */
#include "global.h"
#include "data_interface.h"
#include "utility.h"
#include "lib_io_file.h"
#include "lib_op_line.h"




using namespace std;




// loading the Matrix matrix with file name as char * filename
// for a matrix, the format is straightforward, that we only need to save the matrix as they are
void load_matrix(Matrix & matrix, char * filename)
{
	//==== load data into temporary container
	vector<vector<float>> container_temp;

	char type[10] = "r";
	filehandle file(filename, type);

	long input_length = 1000000000;
	char * line = (char *)malloc( sizeof(char) * input_length );
	while(1)
	{
		int end = file.readline(line, input_length);
		if(end)
			break;

		line_class line_obj(line);
		line_obj.split_tab();
		vector<float> vec;
		for(unsigned i=0; i<line_obj.size(); i++)
		{
			char * pointer = line_obj.at(i);
			//float value = stof(pointer);			// NOTE: there are double-range numbers
			float value = stod(pointer);
			vec.push_back(value);
		}
		line_obj.release();

		container_temp.push_back(vec);
	}
	free(line);
	file.close();


	//==== load data into Matrix from temporary container
	matrix.init(container_temp);

	return;
}



// load a tensor into Tensor tensor, with filename as char * filename
// since I want to keep the tensor as a whole, other than splitting them into sub-files, I will use meta info (first line, shape of tensor)
void load_tensor(Tensor & tensor, char * filename)
{
	char type[10] = "r";
	filehandle file(filename, type);

	long input_length = 1000000000;
	char * line = (char *)malloc( sizeof(char) * input_length );


	//==== first, get tensor shape
	int dimension1 = 0;
	int dimension2 = 0;
	int dimension3 = 0;

	file.readline(line, input_length);
	line_class line_obj(line);
	line_obj.split_tab();
	char * pointer;
	//== d1
	pointer = line_obj.at(0);
	dimension1 = atoi(pointer);
	//== d2
	pointer = line_obj.at(1);
	dimension2 = atoi(pointer);
	//== d3
	pointer = line_obj.at(2);
	dimension3 = atoi(pointer);

	line_obj.release();


	//==== then, load data into temporary container
	vector<vector<vector<float>>> container_temp;

	for(int i=0; i<dimension1; i++)
	{
		vector<vector<float>> vec;
		container_temp.push_back(vec);

		for(int j=0; j<dimension2; j++)
		{
			int end = file.readline(line, input_length);

			line_class line_obj(line);
			line_obj.split_tab();

			vector<float> vec;
			for(unsigned i=0; i<line_obj.size(); i++)
			{
				char * pointer = line_obj.at(i);
				//float value = stof(pointer);				// there are double-range numbers
				float value = stod(pointer);
				vec.push_back(value);
			}
			line_obj.release();
			(container_temp.at(i)).push_back(vec);

		}
	}

	free(line);
	file.close();


	//==== load data into Tensor from temporary container
	tensor.init(container_temp);


	return;
}




//// loading the simulated data
void data_load_simu()
{
	// global:
	//	X, Y, markerset
	//	Y1, U1, Beta, V1, T1, Y2, U2, V2, T2
	//	K, I, J, S, D1, D2
	//	alpha, N_element
	cout << "loading the simu data..." << endl;


	//==========================================
	//==== load parameters, init
	char filename[100];

	//==== matrix
	//== U1
	sprintf(filename, "../data_simu/U1.txt");
	load_matrix(U1, filename);
	//== V1
	sprintf(filename, "../data_simu/V1.txt");
	load_matrix(V1, filename);
	//== T1
	sprintf(filename, "../data_simu/T1.txt");
	load_matrix(T1, filename);
	//== U2
	sprintf(filename, "../data_simu/U2.txt");
	load_matrix(U2, filename);
	//== V2
	sprintf(filename, "../data_simu/V2.txt");
	load_matrix(V2, filename);
	//== T2
	sprintf(filename, "../data_simu/T2.txt");
	load_matrix(T2, filename);
	//== Beta
	sprintf(filename, "../data_simu/Beta.txt");
	load_matrix(Beta, filename);

	//==== tensor
	//== Y1
	sprintf(filename, "../data_simu/Y1.txt");
	load_tensor(Y1, filename);
	//== Y2
	sprintf(filename, "../data_simu/Y2.txt");
	load_tensor(Y2, filename);


	//==========================================
	//==== load data
	//== X
	sprintf(filename, "../data_simu/X.txt");
	load_matrix(X, filename);
	//== Y
	sprintf(filename, "../data_simu/Y.txt");
	load_tensor(Y, filename);


	//==========================================
	//==== fill in the dimensions
	K = Y.get_dimension1();
	I = Y.get_dimension2();
	J = Y.get_dimension3();
	//
	S = Beta.get_dimension2();
	D1 = Beta.get_dimension1();
	//
	D2 = U2.get_dimension2();


	//==========================================
	//==== the others
	markerset.init(K, I, J, 1);
	N_element = int(markerset.sum());
	alpha = 1.0;				// NOTE: need to manually set this


	return;
}




//// loading the real data
void data_load_real()
{
	// global:
	//	X, Y, markerset
	//	Y1, U1, Beta, V1, T1, Y2, U2, V2, T2
	//	K, I, J, S, D1, D2
	//	alpha, markerset, N_element
	cout << "loading the real data..." << endl;

	//==========================================
	//==== load parameters, init
	char filename[100];

	//==== matrix
	//== U1
	sprintf(filename, "../data_real/U1.txt");
	load_matrix(U1, filename);
	//== V1
	sprintf(filename, "../data_real/V1.txt");
	load_matrix(V1, filename);
	//== T1
	sprintf(filename, "../data_real/T1.txt");
	load_matrix(T1, filename);
	//== U2
	sprintf(filename, "../data_real/U2.txt");
	load_matrix(U2, filename);
	//== V2
	sprintf(filename, "../data_real/V2.txt");
	load_matrix(V2, filename);
	//== T2
	sprintf(filename, "../data_real/T2.txt");
	load_matrix(T2, filename);
	//== Beta
	sprintf(filename, "../data_real/Beta.txt");
	load_matrix(Beta, filename);

	//==== tensor
	//== Y1
	sprintf(filename, "../data_real/Y1.txt");
	load_tensor(Y1, filename);
	//== Y2
	sprintf(filename, "../data_real/Y2.txt");
	load_tensor(Y2, filename);


	//==========================================
	//==== fill in the dimensions
	K = Y1.get_dimension1();
	I = Y1.get_dimension2();
	J = Y1.get_dimension3();
	//
	S = Beta.get_dimension2();
	D1 = Beta.get_dimension1();
	//
	D2 = U2.get_dimension2();


	//==========================================
	//==== load data
	//== X
	sprintf(filename, "../data_real/X.txt");
	load_matrix(X, filename);

	// Y (dataset), markerset
	Y.init(K, I, J);
	markerset.init(K, I, J, 0);
	for(int k=0; k<K; k++)
	{
		char filename[100];
		filename[0] = '\0';
		strcat(filename, "../data_real/tensor/Tensor_tissue_");
		char tissue[10];
		sprintf(tissue, "%d", k);
		strcat(filename, tissue);
		strcat(filename, ".txt");

		char type[10] = "r";
		filehandle file(filename, type);

		long input_length = 1000000000;
		char * line = (char *)malloc( sizeof(char) * input_length );
		while(1)
		{
			int end = file.readline(line, input_length);
			if(end)
				break;

			line_class line_obj(line);
			line_obj.split_tab();

			int index = atoi(line_obj.at(0));
			vector<float> vec;
			for(unsigned i=1; i<line_obj.size(); i++)		// NOTE: here we start from pos#1
			{
				char * pointer = line_obj.at(i);
				//float value = stof(pointer);				// NOTE: there are double-range numbers
				float value = stod(pointer);
				vec.push_back(value);
			}
			line_obj.release();
			for(int i=0; i<vec.size(); i++)
			{
				Y.set_element(k, index, i, vec.at(i));
				markerset.set_element(k, index, i, 1);
			}

		}
		free(line);
		file.close();
	}

	cout << "Y and markerset shape:" << endl;
	cout << "(" << Y.get_dimension1() << ", " << Y.get_dimension2() << ", " << Y.get_dimension3() << ")" << endl;
	cout << "(" << markerset.get_dimension1() << ", " << markerset.get_dimension2() << ", " << markerset.get_dimension3() << ")" << endl;


	//==========================================
	//==== init others
	alpha = 1.0;		// just random
	N_element = int(markerset.sum());


	return;
}





// save the learned model
// where: "../result/"
void data_save()
{
	cout << "now saving the learned models... (Y1, Y2, U1, V1, T1, Beta, U2, V2, T2, alpha)" << endl;

	char filename[100];

	//==== matrix
	//== U1
	sprintf(filename, "../result/Y1.txt");
	Y1.save(filename);
	sprintf(filename, "../result/U1.txt");
	U1.save(filename);
	sprintf(filename, "../result/Beta.txt");
	Beta.save(filename);
	sprintf(filename, "../result/V1.txt");
	V1.save(filename);
	sprintf(filename, "../result/T1.txt");
	T1.save(filename);
	sprintf(filename, "../result/Y2.txt");
	Y2.save(filename);
	sprintf(filename, "../result/U2.txt");
	U2.save(filename);
	sprintf(filename, "../result/V2.txt");
	V2.save(filename);
	sprintf(filename, "../result/T2.txt");
	T2.save(filename);


	// NOTE: specially for alpha
	sprintf(filename, "../result/alpha.txt");
	FILE * file_out = fopen(filename, "w+");
	if(file_out == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}
	char buf[1024];
	sprintf(buf, "%f\n", alpha);
	fwrite(buf, sizeof(char), strlen(buf), file_out);
	fclose(file_out);


	return;
}



// init loglike container
void loglike_init()
{
	loglike_total.clear();
	loglike_data.clear();
	loglike_Y1.clear();
	loglike_Y2.clear();
	loglike_U1.clear();
	loglike_V1.clear();
	loglike_T1.clear();
	loglike_U2.clear();
	loglike_V2.clear();
	loglike_T2.clear();
	loglike_Beta.clear();
	loglike_alpha.clear();

	return;
}



void save_vector(vector<float> & vec, char * filename)
{
	FILE * file_out = fopen(filename, "w+");
	if(file_out == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}

	for(int i=0; i<vec.size(); i++)
	{
		float value = vec.at(i);
		char buf[1024];
		sprintf(buf, "%f\n", value);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
	}
	fclose(file_out);

	return;
}



// save loglike per need
// where: "../result/"
void loglike_save()
{
	char filename[] = "../result/loglike_total.txt";
	save_vector(loglike_total, filename);

	return;
}


