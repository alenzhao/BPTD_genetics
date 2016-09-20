## this is used to init the two factor tensors (with their factor matrices)
## the general idea is to use the incomplete PCA to init one and the residual to init the other one


## this script takes the preprocessed and normalized tensor as input
## this script will get accompany with incomplete PCA R script
## this script will be followed by Beta init





## initialize the tensor factor matrices
## targets:
##	Tissue.npy,				in-directly
##	Individual.npy,			in-directly
##	Gene.npy,				directly
##
## and reformat them in .txt







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


n_factor = 400			## TODO: tune; TODO: probably set factor_genetics and factor_nongenetics separately
n_tissue = 0
n_individual = 0
n_gene = 0
dimension = (n_tissue, n_individual, n_gene)



##===============
##==== sub-routines
##===============
# get the "xxx-yyy" from "xxx-yyy-zzz-aaa-qqq", which is defined as the individual ID of the GTEx samples
pattern_indiv = re.compile(r'^(\w)+([\-])(\w)+')
def get_individual_id(s):
	match = pattern_indiv.match(s)
	if match:
		return match.group()
	else:
		print "!!! no individual ID is found..."
		return ""




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
	list_tissue = np.load("./data_prepared/Tissue_list.npy")
	list_individual = np.load("./data_prepared/Individual_list.npy")
	list_gene = np.load("./data_prepared/Gene_list.npy")
	list_sample = np.load("./data_raw/list_sample.npy")
	Data = np.load("./data_raw/Data.npy")
	## sample_tissue_rep
	file = open("./data_raw/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_sample_tissue_type", 'r')
	sample_tissue_rep = {}
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split('\t')
		if len(line) < 3:
			print line
			continue

		sample = line[0]
		tissue = line[2]
		sample_tissue_rep[sample] = tissue
	file.close()


	n_tissue = len(list_tissue)
	n_individual = len(list_individual)
	n_gene = len(list_gene)
	dimension = (n_tissue, n_individual, n_gene)
	print dimension





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

	##==== save PCA results (two matrices, for coefficient matrix and factor matrix; and also the sample_list)
	#np.save("./data_processed/pca_sample", Y1)
	#np.save("./data_processed/pca_gene", Y2)
	#np.save("./data_processed/pca_variance", variance)
	##print "saving the .npy data done..."

	np.save("./data_init/Gene_tf1", Y2)



	##=============
	##==== save the Individual x Tissue matrix (with Nan in) under "./data_inter/"
	##=============
	rep_tissue_index = {}
	for i in range(len(list_tissue)):
		tissue = list_tissue[i]
		rep_tissue_index[tissue] = i
	rep_individual_index = {}
	for i in range(len(list_individual)):
		individual = list_individual[i]
		rep_individual_index[individual] = i

	Data = np.zeros((n_tissue, n_individual, n_factor))
	for i in range(n_tissue):
		for j in range(n_individual):
			for k in range(n_factor):
				Data[(i,j,k)] = float("Nan")
	for i in range(len(list_sample)):
		sample = list_sample[i]
		tissue = sample_tissue_rep[sample]
		index_tissue = rep_tissue_index[tissue]
		individual = get_individual_id(sample)
		index_individual = rep_individual_index[individual]

		Data[index_tissue][index_individual] = Y1[i]

	print "the Tissue x Individual x Factor tensor has the dimension:",
	print Data.shape


	for k in range(n_factor):
		m_factor = np.zeros((n_tissue, n_individual))
		for i in range(n_tissue):
			for j in range(n_individual):
				m_factor[i][j] = Data[i][j][k]
		np.save("./data_inter/f" + str(k) + "_tissue_indiv", m_factor)
	print "save done..."


	##== need to save the results in tsv file (including Nan), in order to load in R
	for k in range(n_factor):
		m = np.load("./data_inter/f" + str(k) + "_tissue_indiv.npy")
		file = open("./data_inter/f" + str(k) + "_tissue_indiv.txt", 'w')
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







	"""
	######## SESSION II ########
	factor_tissue = []
	factor_indiv = []
	for k in range(n_factor):
		#
		factor_tissue.append([])
		file = open("./data_inter/f" + str(k) + "_tissue.txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break
			factor_tissue[-1].append(float(line))
		file.close()

		#
		factor_indiv.append([])
		file = open("./data_inter/f" + str(k) + "_indiv.txt", 'r')
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

	np.save("./data_init/Tissue_tf1", factor_tissue)
	np.save("./data_init/Individual_tf1", factor_indiv)


	##==== save tensor factor
	fm1 = np.load("./data_init/Tissue_tf1.npy")
	fm2 = np.load("./data_init/Individual_tf1.npy")
	fm3 = np.load("./data_init/Gene_tf1.npy")
	TF1 = tensor_cal(fm1, fm2, fm3)
	np.save("./data_init/TF1", TF1)
	reformat_tensor(TF1, "./data_init/TF1.txt")
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
	list_tissue = np.load("./data_prepared/Tissue_list.npy")
	list_individual = np.load("./data_prepared/Individual_list.npy")
	list_gene = np.load("./data_prepared/Gene_list.npy")
	list_sample = np.load("./data_raw/list_sample.npy")
	Data = np.load("./data_raw/Data.npy")
	## sample_tissue_rep
	file = open("./data_raw/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_sample_tissue_type", 'r')
	sample_tissue_rep = {}
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split('\t')
		if len(line) < 3:
			print line
			continue

		sample = line[0]
		tissue = line[2]
		sample_tissue_rep[sample] = tissue
	file.close()
	#
	n_tissue = len(list_tissue)
	n_individual = len(list_individual)
	n_gene = len(list_gene)
	dimension = (n_tissue, n_individual, n_gene)
	print dimension

	#
	rep_tissue_index = {}
	for i in range(len(list_tissue)):
		tissue = list_tissue[i]
		rep_tissue_index[tissue] = i
	rep_individual_index = {}
	for i in range(len(list_individual)):
		individual = list_individual[i]
		rep_individual_index[individual] = i

	TF1 = np.load("./data_init/TF1.npy")
	TF1_matrix = np.zeros(Data.shape)
	for i in range(len(list_sample)):
		sample = list_sample[i]
		tissue = sample_tissue_rep[sample]
		index_tissue = rep_tissue_index[tissue]
		individual = get_individual_id(sample)
		index_individual = rep_individual_index[individual]

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

	##==== save PCA results (two matrices, for coefficient matrix and factor matrix; and also the sample_list)
	#np.save("./data_processed/pca_sample", Y1)
	#np.save("./data_processed/pca_gene", Y2)
	#np.save("./data_processed/pca_variance", variance)
	##print "saving the .npy data done..."

	np.save("./data_init/Gene_tf2", Y2)


	##=============
	##==== save the Individual x Tissue matrix (with Nan in) under "./data_inter/"
	##=============
	rep_tissue_index = {}
	for i in range(len(list_tissue)):
		tissue = list_tissue[i]
		rep_tissue_index[tissue] = i
	rep_individual_index = {}
	for i in range(len(list_individual)):
		individual = list_individual[i]
		rep_individual_index[individual] = i

	Data = np.zeros((n_tissue, n_individual, n_factor))
	for i in range(n_tissue):
		for j in range(n_individual):
			for k in range(n_factor):
				Data[(i,j,k)] = float("Nan")
	for i in range(len(list_sample)):
		sample = list_sample[i]
		tissue = sample_tissue_rep[sample]
		index_tissue = rep_tissue_index[tissue]
		individual = get_individual_id(sample)
		index_individual = rep_individual_index[individual]

		Data[index_tissue][index_individual] = Y1[i]

	print "the Tissue x Individual x Factor tensor has the dimension:",
	print Data.shape


	for k in range(n_factor):
		m_factor = np.zeros((n_tissue, n_individual))
		for i in range(n_tissue):
			for j in range(n_individual):
				m_factor[i][j] = Data[i][j][k]
		np.save("./data_inter/f" + str(k) + "_tissue_indiv", m_factor)
	print "save done..."


	##== need to save the results in tsv file (including Nan), in order to load in R
	for k in range(n_factor):
		m = np.load("./data_inter/f" + str(k) + "_tissue_indiv.npy")
		file = open("./data_inter/f" + str(k) + "_tissue_indiv.txt", 'w')
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
	factor_tissue = []
	factor_indiv = []
	for k in range(n_factor):
		#
		factor_tissue.append([])
		file = open("./data_inter/f" + str(k) + "_tissue.txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break
			factor_tissue[-1].append(float(line))
		file.close()

		#
		factor_indiv.append([])
		file = open("./data_inter/f" + str(k) + "_indiv.txt", 'r')
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

	np.save("./data_init/Tissue_tf2", factor_tissue)
	np.save("./data_init/Individual_tf2", factor_indiv)


	##==== save tensor factor
	fm1 = np.load("./data_init/Tissue_tf2.npy")
	fm2 = np.load("./data_init/Individual_tf2.npy")
	fm3 = np.load("./data_init/Gene_tf2.npy")
	TF2 = tensor_cal(fm1, fm2, fm3)
	np.save("./data_init/TF2", TF2)
	reformat_tensor(TF2, "./data_init/TF2.txt")








	print "done..."






