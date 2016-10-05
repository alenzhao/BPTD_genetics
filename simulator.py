## simulate from real genotype data and the model
## simulation is for testing purpose (convergence)




##==== libraries
import math
import numpy as np




##==== global variables
X = []
Y = []
Y1 = []
U1 = []
V1 = []
T1 = []
Beta = []
Y2 = []
U2 = []
V2 = []
T2 = []
alpha = 1




'''
# NOTE: new whole-genome data scale
K = 28
I = 449
J = 19425
S = 158617
D1 = 200
D2 = 200
'''



'''
# NOTE: the following are the real scenario for chr22 and brain tensor
K = 13
I = 159
J = 585
S = 14056				#S = 14056
D1 = 40
D2 = 40
'''



'''
# NOTE: the following are for the whole-genome and all samples (old)
K = 33
I = 450
J = 21150
S = 824113
D1 = 400
D2 = 400
'''



# NOTE: 10% of real scale
K = 13
I = 159
J = 2000
S = 100000
D1 = 40
D2 = 40






##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============



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


##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============
##=============/=============/=============/=============/=============/=============/=============/=============




##== func: simulate a (d_sample x d_feature) length factor matrix (np array)
##== for now, it's as simple as a uni-variate Gaussian, for all the elements
##== we can change this module later on, for more structured factor matrix
def simu_fm_uniform(d_sample, d_feature, mu, lamb):		## (local function)
	sigma = math.sqrt(1.0 / lamb)

	##== simu
	fm = np.random.normal(mu, sigma, (d_sample, d_feature))
	return fm



## sample uni-Gaussian from given mean matrix (np.array) and variance
def simu_fm_Mu(d_sample, d_feature, Mu, lamb):			## (local function)
	sigma = math.sqrt(1.0 / lamb)

	##== simu
	fm = np.zeros((d_sample, d_feature))
	for i in range(d_sample):
		for j in range(d_feature):
			mu = Mu[(i,j)]
			fm[(i,j)] = np.random.normal(mu, sigma)
	return fm



def simu_Y1(lamb, mu1, lamb1):		# (work on global scale)
	global Y1, U1, I, V1, J, T1, K, D1, Beta, S, X

	##== simu priors
	Beta = simu_fm_uniform(D1, S, mu1, lamb1)



	## NOTE: we tune the power to be small here
	#Beta = Beta / 100



	Mu = np.zeros((I, D1))
	for i in range(I):
		for d in range(D1):
			mean = np.inner(X[i], Beta[d])
			Mu[(i,d)] = mean
	U1 = simu_fm_Mu(I, D1, Mu, lamb)
	V1 = simu_fm_uniform(J, D1, mu1, lamb1)
	T1 = simu_fm_uniform(K, D1, mu1, lamb1)

	##== simu Y1
	sigma = math.sqrt(1.0 / lamb)
	Y1 = np.zeros((K, I, J))
	for k in range(K):
		for i in range(I):
			for j in range(J):
				temp = np.multiply(U1[i], V1[j])
				mean = np.inner(temp, T1[k])
				draw = np.random.normal(mean, sigma)
				Y1[(k,i,j)] = draw
	return



def simu_Y2(lamb, mu1, lamb1):		# (work on global scale)
	global Y2, U2, I, V2, J, T2, K, D2

	##== simu priors
	U2 = simu_fm_uniform(I, D2, mu1, lamb1)
	V2 = simu_fm_uniform(J, D2, mu1, lamb1)
	T2 = simu_fm_uniform(K, D2, mu1, lamb1)

	##== simu Y2
	sigma = math.sqrt(1.0 / lamb)
	Y2 = np.zeros((K, I, J))
	for k in range(K):
		for i in range(I):
			for j in range(J):
				temp = np.multiply(U2[i], V2[j])
				mean = np.inner(temp, T2[k])
				draw = np.random.normal(mean, sigma)
				Y2[(k,i,j)] = draw
	return




def simu_alpha(alpha0, beta0):		# (work on global scale)
	global alpha

	shape = alpha0
	scale = 1.0 / beta0

	alpha = np.random.gamma(shape, scale)
	return




def simu_Y():						# (work on global scale)
	global Y, I, J, K, Y1, Y2, alpha
	sigma = math.sqrt(1.0 / alpha)

	##== simu Y
	Y = np.zeros((K, I, J))
	for k in range(K):
		for i in range(I):
			for j in range(J):
				mean = Y1[(k,i,j)] + Y2[(k,i,j)]
				draw = np.random.normal(mean, sigma)
				Y[(k,i,j)] = draw
	return





if __name__ == "__main__":


	print "this is simulator..."


	"""
	##==== load X (for brain and chr22)
	list_indiv = np.load("../Individual_list.npy")

	X = []
	for indiv in list_indiv:
		file = open("../genotype_450_dosage_matrix_qc/chr22/SNP_dosage_" + indiv + ".txt", 'r')
		X.append([])
		while 1:
			line = (file.readline()).strip()
			if not line:
				break

			dosage = float(line)
			X[-1].append(dosage)
		file.close()
	X = np.array(X)
	X = X[:I]

	print "dosage matrix shape:",
	print X.shape
	"""


	##==== load X (for whole genome and all samples)
	list_indiv = np.load("../Individual_list.npy")
	print "# of individuals:",
	print len(list_indiv)

	X = []
	for indiv in list_indiv:
		X.append([])

		for i in range(22):
			chr = i+1
			file = open("../genotype_450_dosage_matrix_qc_trim/chr" + str(chr) + "/SNP_dosage_" + indiv + ".txt", 'r')	## NOTE: on cluster
			while 1:
				line = (file.readline()).strip()
				if not line:
					break

				dosage = float(line)
				X[-1].append(dosage)
			file.close()

	X = np.array(X)

	print "dosage matrix shape:",
	print X.shape

	## pick up a subset of all sites
	X = X[:I, :S]
	print "the subset X size:",
	print X.shape






	##========================================================================================================
	## simulating the data
	##========================================================================================================
	print "[@@@] now simulating the real data..."
	##==== simulation
	## I will treat all observed variables as input of simu functions
	mu = 0						## TODO
	lamb = 1.0					## TODO
	alpha0 = 1.0				## TODO
	beta0 = 0.5					## TODO

	simu_Y1(lamb, mu, lamb)
	simu_Y2(lamb, mu, lamb)
	simu_alpha(alpha0, beta0)
	simu_Y()


	print "simulation done..."


	##==== saving
	print "now saving the simulated data and model..."

	print "X shape:", X.shape
	np.save("./data_simu/X", X)
	print "Y shape:", Y.shape
	np.save("./data_simu/Y", Y)
	print "Y1 shape:", Y1.shape
	np.save("./data_simu/Y1", Y1)
	print "U1 shape:", U1.shape
	np.save("./data_simu/U1", U1)
	print "Beta shape:", Beta.shape
	np.save("./data_simu/Beta", Beta)
	print "V1 shape:", V1.shape
	np.save("./data_simu/V1", V1)
	print "T1 shape:", T1.shape
	np.save("./data_simu/T1", T1)
	print "Y2 shape:", Y2.shape
	np.save("./data_simu/Y2", Y2)
	print "U2 shape:", U2.shape
	np.save("./data_simu/U2", U2)
	print "V2 shape:", V2.shape
	np.save("./data_simu/V2", V2)
	print "T2 shape:", T2.shape
	np.save("./data_simu/T2", T2)
	np.save("./data_simu/alpha", np.array([alpha]))




	##==== re-saving everything in .txt format
	#==== matrix
	#
	fm = np.load("./data_simu/T1.npy")
	reformat_matrix(fm, "./data_simu/T1.txt")
	fm = np.load("./data_simu/U1.npy")
	reformat_matrix(fm, "./data_simu/U1.txt")
	fm = np.load("./data_simu/V1.npy")
	reformat_matrix(fm, "./data_simu/V1.txt")

	#
	fm = np.load("./data_simu/T2.npy")
	reformat_matrix(fm, "./data_simu/T2.txt")
	fm = np.load("./data_simu/U2.npy")
	reformat_matrix(fm, "./data_simu/U2.txt")
	fm = np.load("./data_simu/V2.npy")
	reformat_matrix(fm, "./data_simu/V2.txt")

	#
	fm = np.load("./data_simu/Beta.npy")
	reformat_matrix(fm, "./data_simu/Beta.txt")

	#
	X = np.load("./data_simu/X.npy")
	reformat_matrix(X, "./data_simu/X.txt")


	#==== tensor
	Y = np.load("./data_simu/Y.npy")
	reformat_tensor(Y, "./data_simu/Y.txt")
	Y1 = np.load("./data_simu/Y1.npy")
	reformat_tensor(Y1, "./data_simu/Y1.txt")
	Y2 = np.load("./data_simu/Y2.npy")
	reformat_tensor(Y2, "./data_simu/Y2.txt")







	##========================================================================================================
	## simulating another copy of model
	##========================================================================================================
	print "[@@@] now simulating another copy of the model..."
	##==== simulation
	simu_Y1(lamb, mu, lamb)
	simu_Y2(lamb, mu, lamb)
	simu_alpha(alpha0, beta0)

	print "simulation (another copy) done..."

	##==== saving
	print "now saving the simulated model..."
	print "Y1 shape:", Y1.shape
	np.save("./data_simu_init/Y1", Y1)
	print "U1 shape:", U1.shape
	np.save("./data_simu_init/U1", U1)
	print "Beta shape:", Beta.shape
	np.save("./data_simu_init/Beta", Beta)
	print "V1 shape:", V1.shape
	np.save("./data_simu_init/V1", V1)
	print "T1 shape:", T1.shape
	np.save("./data_simu_init/T1", T1)
	print "Y2 shape:", Y2.shape
	np.save("./data_simu_init/Y2", Y2)
	print "U2 shape:", U2.shape
	np.save("./data_simu_init/U2", U2)
	print "V2 shape:", V2.shape
	np.save("./data_simu_init/V2", V2)
	print "T2 shape:", T2.shape
	np.save("./data_simu_init/T2", T2)
	np.save("./data_simu_init/alpha", np.array([alpha]))


	##==== re-saving everything in .txt format
	#==== matrix
	#
	fm = np.load("./data_simu_init/T1.npy")
	reformat_matrix(fm, "./data_simu_init/T1.txt")
	fm = np.load("./data_simu_init/U1.npy")
	reformat_matrix(fm, "./data_simu_init/U1.txt")
	fm = np.load("./data_simu_init/V1.npy")
	reformat_matrix(fm, "./data_simu_init/V1.txt")

	#
	fm = np.load("./data_simu_init/T2.npy")
	reformat_matrix(fm, "./data_simu_init/T2.txt")
	fm = np.load("./data_simu_init/U2.npy")
	reformat_matrix(fm, "./data_simu_init/U2.txt")
	fm = np.load("./data_simu_init/V2.npy")
	reformat_matrix(fm, "./data_simu_init/V2.txt")

	#
	fm = np.load("./data_simu_init/Beta.npy")
	reformat_matrix(fm, "./data_simu_init/Beta.txt")

	#==== tensor
	Y1 = np.load("./data_simu_init/Y1.npy")
	reformat_tensor(Y1, "./data_simu_init/Y1.txt")
	Y2 = np.load("./data_simu_init/Y2.npy")
	reformat_tensor(Y2, "./data_simu_init/Y2.txt")









