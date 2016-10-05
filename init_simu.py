##
## we simulate the full tensor, and now we'll make it incomplete, and initialize as we will for the real data
##

## NOTE: mimic init_1/2/3 for real data
## NOTE: the format of sample ID: tissueID-individualPos







##===============
##==== libraries
##===============
import numpy as np
from numpy.linalg import inv
from scipy.stats import wishart
import math
from numpy import linalg as LA
from copy import *
import cProfile
import timeit
from sklearn.decomposition import PCA
import re


n_factor = 40			## TODO: tune; TODO: probably set factor_genetics and factor_nongenetics separately
n_tissue = 0
n_individual = 0
n_gene = 0
dimension = (n_tissue, n_individual, n_gene)




##===============
##==== sub-routines
##===============
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



def tensor_cal(T, U, V):
	dimension1 = len(T)
	dimension2 = len(U)
	dimension3 = len(V)
	tensor1 = []
	for i in range(dimension2):
		tensor1.append([])
		for j in range(dimension3):
			array = np.multiply(U[i], V[j])
			tensor1[-1].append(array)
	tensor = []
	for k in range(dimension1):
		tensor.append([])
		for i in range(dimension2):
			tensor[-1].append([])
			for j in range(dimension3):
				value = np.inner(tensor1[i][j], T[k])
				tensor[-1][-1].append(value)
	tensor = np.array(tensor)
	return tensor







if __name__ == "__main__":






	######## SESSION I -- one factor tensor ########
	"""
	print "working on tf1..."
	##=============
	##==== loading
	##=============
	## dimension prepare
	U1 = np.load("./data_simu_init/U1.npy")
	V1 = np.load("./data_simu_init/V1.npy")
	T1 = np.load("./data_simu_init/T1.npy")
	n_factor = len(U1[0])
	n_tissue = len(T1)
	n_individual = len(U1)
	n_gene = len(V1)
	dimension = (n_tissue, n_individual, n_gene)
	print "real tensor shape:",
	print dimension


	## load data
	list_sample = []
	Data = []
	for k in range(n_tissue):
		list_sample_tissue = np.load("./data_simu_init/Tensor_tissue_" + str(k) + "_list.npy")
		Data_tissue = np.load("./data_simu_init/Tensor_tissue_" + str(k) + ".npy")
		for i in range(len(list_sample_tissue)):
			sample = list_sample_tissue[i]
			data = Data_tissue[i]
			list_sample.append(sample)
			Data.append(data)
	list_sample = np.array(list_sample)
	Data = np.array(Data)



	##=============
	##==== do PCA for Sample x Gene matrix
	##=============
	print "performing PCA..."
	pca = PCA(n_components=n_factor)
	pca.fit(Data)
	Y2 = (pca.components_).T
	Y1 = pca.transform(Data)
	variance = pca.explained_variance_ratio_

	print variance
	print "and the cumulative variance are:"
	for i in range(len(variance)):
		print i,
		print np.sum(variance[:i+1]),
	print ""

	# DEBUG
	print "sample factor matrix:",
	print len(Y1),
	print len(Y1[0])
	print "gene factor matrix:",
	print len(Y2),
	print len(Y2[0])
	np.save("./data_simu_init/Gene_tf1", Y2)


	##=============
	##==== save the Individual x Tissue matrix (with Nan in) under "./data_inter/"
	##=============
	Data = np.zeros((n_tissue, n_individual, n_factor))
	for i in range(n_tissue):
		for j in range(n_individual):
			for k in range(n_factor):
				Data[(i,j,k)] = float("Nan")
	for i in range(len(list_sample)):
		sample = list_sample[i]
		pair = sample.split('-')
		tissue = pair[0]
		index_tissue = int(tissue)
		individual = pair[1]
		index_individual = int(individual)
		#
		Data[index_tissue][index_individual] = Y1[i]

	print "the Tissue x Individual x Factor tensor has the dimension:",
	print Data.shape


	for k in range(n_factor):
		m_factor = np.zeros((n_tissue, n_individual))
		for i in range(n_tissue):
			for j in range(n_individual):
				m_factor[i][j] = Data[i][j][k]
		np.save("./data_simu_inter/f" + str(k) + "_tissue_indiv", m_factor)
	print "save done..."


	##== need to save the results in tsv file (including Nan), in order to load in R
	for k in range(n_factor):
		m = np.load("./data_simu_inter/f" + str(k) + "_tissue_indiv.npy")
		file = open("./data_simu_inter/f" + str(k) + "_tissue_indiv.txt", 'w')
		for i in range(len(m)):
			for j in range(len(m[i])):
				value = m[i][j]
				file.write(str(value))
				if j != len(m[i])-1:
					file.write('\t')
			file.write('\n')
		file.close()
	"""









	###########################################
	###########################################
	###########################################
	######## R code for incomplete CPA ########
	###########################################
	###########################################
	###########################################








	######## SESSION II ########
	"""
	factor_tissue = []
	factor_indiv = []
	for k in range(n_factor):
		#
		factor_tissue.append([])
		file = open("./data_simu_inter/f" + str(k) + "_tissue.txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break
			factor_tissue[-1].append(float(line))
		file.close()

		#
		factor_indiv.append([])
		file = open("./data_simu_inter/f" + str(k) + "_indiv.txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break
			factor_indiv[-1].append(float(line))
		file.close()

	factor_tissue = (np.array(factor_tissue)).T
	factor_indiv = (np.array(factor_indiv)).T

	print "factor tissue:",
	print factor_tissue.shape

	print "factor indiv:",
	print factor_indiv.shape

	np.save("./data_simu_init/Tissue_tf1", factor_tissue)
	np.save("./data_simu_init/Individual_tf1", factor_indiv)


	##==== save tensor factor
	fm1 = np.load("./data_simu_init/Tissue_tf1.npy")
	fm2 = np.load("./data_simu_init/Individual_tf1.npy")
	fm3 = np.load("./data_simu_init/Gene_tf1.npy")
	TF1 = tensor_cal(fm1, fm2, fm3)
	np.save("./data_simu_init/TF1", TF1)
	reformat_tensor(TF1, "./data_simu_init/TF1.txt")
	"""











	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============









	######## INTER SESSION ########
	"""
	## purpose:
	##	1. get the new Data, which is the residuals of Tensor after tensor factor #1
	#
	## dimension prepare
	U1 = np.load("./data_simu_init/U1.npy")
	V1 = np.load("./data_simu_init/V1.npy")
	T1 = np.load("./data_simu_init/T1.npy")
	n_factor = len(U1[0])
	n_tissue = len(T1)
	n_individual = len(U1)
	n_gene = len(V1)
	dimension = (n_tissue, n_individual, n_gene)
	print "real tensor shape:",
	print dimension

	## load data
	list_sample = []
	Data = []
	for k in range(n_tissue):
		list_sample_tissue = np.load("./data_simu_init/Tensor_tissue_" + str(k) + "_list.npy")
		Data_tissue = np.load("./data_simu_init/Tensor_tissue_" + str(k) + ".npy")
		for i in range(len(list_sample_tissue)):
			sample = list_sample_tissue[i]
			data = Data_tissue[i]
			list_sample.append(sample)
			Data.append(data)
	list_sample = np.array(list_sample)
	Data = np.array(Data)

	##
	TF1 = np.load("./data_simu_init/TF1.npy")
	TF1_matrix = np.zeros(Data.shape)
	for i in range(len(list_sample)):
		sample = list_sample[i]
		pair = sample.split('-')
		tissue = pair[0]
		index_tissue = int(tissue)
		individual = pair[1]
		index_individual = int(individual)
		#
		TF1_matrix[i] = TF1[index_tissue][index_individual]

	TF1_matrix = np.array(TF1_matrix)
	print "TF1_matrix has shape:",
	print TF1_matrix.shape

	# get the residual
	Data = Data - TF1_matrix





	######## SESSION I -- another factor tensor ########
	print "working on tf2..."
	##=============
	##==== do PCA for Sample x Gene matrix
	##=============
	print "performing PCA..."
	pca = PCA(n_components=n_factor)
	pca.fit(Data)
	Y2 = (pca.components_).T
	Y1 = pca.transform(Data)
	variance = pca.explained_variance_ratio_

	print variance
	print "and the cumulative variance are:"
	for i in range(len(variance)):
		print i,
		print np.sum(variance[:i+1]),
	print ""

	# DEBUG
	print "sample factor matrix:",
	print len(Y1),
	print len(Y1[0])
	print "gene factor matrix:",
	print len(Y2),
	print len(Y2[0])
	np.save("./data_simu_init/Gene_tf2", Y2)


	##=============
	##==== save the Individual x Tissue matrix (with Nan in) under "./data_inter/"
	##=============
	Data = np.zeros((n_tissue, n_individual, n_factor))
	for i in range(n_tissue):
		for j in range(n_individual):
			for k in range(n_factor):
				Data[(i,j,k)] = float("Nan")
	for i in range(len(list_sample)):
		sample = list_sample[i]
		pair = sample.split('-')
		tissue = pair[0]
		index_tissue = int(tissue)
		individual = pair[1]
		index_individual = int(individual)
		#
		Data[index_tissue][index_individual] = Y1[i]

	print "the Tissue x Individual x Factor tensor has the dimension:",
	print Data.shape


	for k in range(n_factor):
		m_factor = np.zeros((n_tissue, n_individual))
		for i in range(n_tissue):
			for j in range(n_individual):
				m_factor[i][j] = Data[i][j][k]
		np.save("./data_simu_inter/f" + str(k) + "_tissue_indiv", m_factor)
	print "save done..."


	##== need to save the results in tsv file (including Nan), in order to load in R
	for k in range(n_factor):
		m = np.load("./data_simu_inter/f" + str(k) + "_tissue_indiv.npy")
		file = open("./data_simu_inter/f" + str(k) + "_tissue_indiv.txt", 'w')
		for i in range(len(m)):
			for j in range(len(m[i])):
				value = m[i][j]
				file.write(str(value))
				if j != len(m[i])-1:
					file.write('\t')
			file.write('\n')
		file.close()
	"""










	###########################################
	###########################################
	###########################################
	######## R code for incomplete CPA ########
	###########################################
	###########################################
	###########################################









	######## SESSION II ########
	"""
	factor_tissue = []
	factor_indiv = []
	for k in range(n_factor):
		#
		factor_tissue.append([])
		file = open("./data_simu_inter/f" + str(k) + "_tissue.txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break
			factor_tissue[-1].append(float(line))
		file.close()

		#
		factor_indiv.append([])
		file = open("./data_simu_inter/f" + str(k) + "_indiv.txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break
			factor_indiv[-1].append(float(line))
		file.close()

	factor_tissue = (np.array(factor_tissue)).T
	factor_indiv = (np.array(factor_indiv)).T

	print "factor tissue:",
	print factor_tissue.shape

	print "factor indiv:",
	print factor_indiv.shape

	np.save("./data_simu_init/Tissue_tf2", factor_tissue)
	np.save("./data_simu_init/Individual_tf2", factor_indiv)


	##==== save tensor factor
	fm1 = np.load("./data_simu_init/Tissue_tf2.npy")
	fm2 = np.load("./data_simu_init/Individual_tf2.npy")
	fm3 = np.load("./data_simu_init/Gene_tf2.npy")
	TF2 = tensor_cal(fm1, fm2, fm3)
	np.save("./data_simu_init/TF2", TF2)
	reformat_tensor(TF2, "./data_simu_init/TF2.txt")






	print "done (kind of)..."
	"""










	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	## do the Beta
	"""
	# U + X -> Beta: X * Beta_inv = U
	X = np.load("./data_simu/X.npy")
	U = np.load("./data_simu_init/Individual_tf1.npy")
	Beta = np.linalg.lstsq(X, U)[0].T
	print "Beta shape:", Beta.shape
	np.save("./data_simu_init/Beta", Beta)
	"""








	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	##=============/=============/=============/=============/=============/=============/=============/=============
	## transfrom from .npy to .txt
	#
	fm = np.load("./data_simu_init/Tissue_tf1.npy")
	reformat_matrix(fm, "./data_simu_init/T1.txt")
	fm = np.load("./data_simu_init/Individual_tf1.npy")
	reformat_matrix(fm, "./data_simu_init/U1.txt")
	fm = np.load("./data_simu_init/Gene_tf1.npy")
	reformat_matrix(fm, "./data_simu_init/V1.txt")

	#
	fm = np.load("./data_simu_init/Tissue_tf2.npy")
	reformat_matrix(fm, "./data_simu_init/T2.txt")
	fm = np.load("./data_simu_init/Individual_tf2.npy")
	reformat_matrix(fm, "./data_simu_init/U2.txt")
	fm = np.load("./data_simu_init/Gene_tf2.npy")
	reformat_matrix(fm, "./data_simu_init/V2.txt")

	#
	fm = np.load("./data_simu_init/Beta.npy")
	reformat_matrix(fm, "./data_simu_init/Beta.txt")











