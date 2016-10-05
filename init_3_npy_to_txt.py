## we reformat the data (X and Y) into .txt, but not yet for some of the model (U1, V1, T1, U2, V2, T2), thus we do that here



import numpy as np



## prepare the numpy input data (and init parameters) into txt format such that C++ can use
"""
The format of the input/output data:
1. Everything in Matrix format should be in matrix format
2. The tensor has one line meta information, the shape of the tensor, and then the tensor is spread as a matrix (collapsed the first dimension)
3. I will also re-prepare the incomplete tensor such that there are only numbers: integer-indexing tissues as usual; inside each tissue, integer indexing individual positions to have this expression sample (one extra integer at the head of each line)
"""
def reformat_matrix(matrix, filename):

	shape = matrix.shape
	dimension1 = shape[0]
	dimension2 = shape[1]

	file = open(filename, 'w')
	for i in range(dimension1):
		for j in range(dimension2):
			value = matrix[i][j]
			file.write(str(value) + '\t')
		file.write('\n')
	file.close()

	return


def reformat_tensor(tensor, filename):

	shape = tensor.shape
	dimension1 = shape[0]
	dimension2 = shape[1]
	dimension3 = shape[2]

	file = open(filename, 'w')
	file.write(str(dimension1) + '\t' + str(dimension2) + '\t' + str(dimension3) + '\n')
	for i in range(dimension1):
		for j in range(dimension2):

			for count in range(dimension3):
				value = tensor[i][j][count]
				file.write(str(value) + '\t')
			file.write('\n')
	file.close()

	return





if __name__=="__main__":



	#
	fm = np.load("./data_init/Tissue_tf1.npy")
	reformat_matrix(fm, "./data_init/T1.txt")
	fm = np.load("./data_init/Individual_tf1.npy")
	reformat_matrix(fm, "./data_init/U1.txt")
	fm = np.load("./data_init/Gene_tf1.npy")
	reformat_matrix(fm, "./data_init/V1.txt")

	#
	fm = np.load("./data_init/Tissue_tf2.npy")
	reformat_matrix(fm, "./data_init/T2.txt")
	fm = np.load("./data_init/Individual_tf2.npy")
	reformat_matrix(fm, "./data_init/U2.txt")
	fm = np.load("./data_init/Gene_tf2.npy")
	reformat_matrix(fm, "./data_init/V2.txt")

	#
	fm = np.load("./data_init/Beta.npy")
	reformat_matrix(fm, "./data_init/Beta.txt")







