## I'm simulating one tensor

## this simulates a symmetric tensor, whose factors are all with uniform Gaussian's



##==== libraries
import math
import numpy as np



##==== global variables
Y = []
U = []
V = []
T = []
alpha = 1
# NOTE: the following are the real scenario for chr22 and brain tensor
K = 13
I = 159
J = 585
D = 40






##== func: simulate a (d_sample x d_feature) length factor matrix (np array)
##== for now, it's as simple as a uni-variate Gaussian, for all the elements
##== we can change this module later on, for more structured factor matrix
def simu_fm_uniform(d_sample, d_feature, mu, lamb):		## (local function)
	sigma = math.sqrt(1.0 / lamb)

	##== simu
	fm = np.random.normal(mu, sigma, (d_sample, d_feature))
	return fm




def simu_Y(mu1, lamb1):		# (work on global scale)
	global Y, U, I, V, J, T, K, D, alpha

	##== simu priors
	U = simu_fm_uniform(I, D, mu1, lamb1)
	V = simu_fm_uniform(J, D, mu1, lamb1)
	T = simu_fm_uniform(K, D, mu1, lamb1)

	##== simu Y1
	sigma = math.sqrt(1.0 / alpha)
	Y = np.zeros((K, I, J))
	for k in range(K):
		for i in range(I):
			for j in range(J):
				temp = np.multiply(U[i], V[j])
				mean = np.inner(temp, T[k])
				draw = np.random.normal(mean, sigma)
				Y[(k,i,j)] = draw
	return





def simu_alpha(alpha0, beta0):		# (work on global scale)
	global alpha

	shape = alpha0
	scale = 1.0 / beta0
	alpha = np.random.gamma(shape, scale)
	return




if __name__ == "__main__":


	print "this is simulator..."

	##==== simulation
	## I will treat all observed variables as input of simu functions
	mu = 0						## TODO
	lamb = 1.0					## TODO
	alpha0 = 1.0				## TODO
	beta0 = 0.5					## TODO


	simu_alpha(alpha0, beta0)
	simu_Y(mu, lamb)

	print "simulation done..."




	##==== saving
	print "now saving the simulated data and model..."
	print "Y shape:", Y.shape
	np.save("./data_simu/Y", Y)
	print "U shape:", U.shape
	np.save("./data_simu/U", U)
	print "V shape:", V.shape
	np.save("./data_simu/V", V)
	print "T shape:", T.shape
	np.save("./data_simu/T", T)
	np.save("./data_simu/alpha", np.array([alpha]))






