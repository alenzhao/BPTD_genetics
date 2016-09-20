## this is used to reformat the SNP data, and init the SNP/genetic factor Beta, per the genetic fm and its order of individuals
## NOTE: it's critical to initialize this parameter matrix appropriately



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





if __name__=="__main__":




	##==== X (genotype)
	list_individual = np.load("./data_prepared/Individual_list.npy")
	X = []
	for individual in list_individual:
		X.append([])
		for chr_index in range(22):
			chr = chr_index + 1
			file = open("../genotype_450_dosage_matrix_qc/chr" + str(chr) + "/SNP_dosage_" + individual + ".txt", 'r')
			while 1:
				line = (file.readline()).strip()
				if not line:
					break

				X[-1].append(float(line))
			file.close()
	X = np.array(X)
	print "X (dosage) shape:", X.shape

	##== reformat data
	reformat_matrix(X, "./data_prepared/X.txt")






	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============






	##==== init model
	# U + X -> Beta: X * Beta_inv = U
	U = np.load("./data_init/Individual_tf1.npy")
	model = np.linalg.lstsq(X, U)
	Beta = np.transpose(model[0])
	print "Beta shape:", Beta.shape

	np.save("./data_init/Beta", Beta)








