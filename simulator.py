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
# NOTE: the following are the real scenario for chr22 and brain tensor
K = 13
I = 159
J = 585
S = 14056
D1 = 40
D2 = 40





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

	##==== load X
	list_indiv = np.load("./data_real/Individual_list.npy")

	X = []
	for indiv in list_indiv:
		file = open("./data_real/genotype_450_dosage_matrix_qc/chr22/SNP_dosage_" + indiv + ".txt", 'r')
		X.append([])
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




