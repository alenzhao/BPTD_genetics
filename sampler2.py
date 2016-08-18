## this samples the data from simulator2.py: one single tensor

## simulate from real genotype data and the model
## simulation is for testing purpose (convergence)

# TODO: didn't implement the incomplete tensor



##==== libraries
import math
import numpy as np
import scipy.special as sps
import timeit




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


loglike_total = []
loglike_Y = []
loglike_U = []
loglike_V = []
loglike_T = []
loglike_alpha = []





##=================
## sampler
##=================
"""
def LaplaceApprox_3D(x, data, mean1, coef, lamb1, mean2, lamb2):
	## likelihood: x, data, mean1, coef, lamb1
	## prior: mean2, lamb2

	##==== pre-cal
	lamb = lamb1 * np.sum( np.power(coef, 2) ) + lamb2
	mean = lamb1 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb2 * mean2
	mean = mean / lamb

	##==== sampler
	sigma = math.sqrt(1.0 / lamb)
	mu = mean
	draw = np.random.normal(mu, sigma)
	return draw
"""



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





def sampler_factor(lamb0, mu1, lamb1):						## (global function)
	global Y, T, U, V, K, I, J, D


	##==== sample the others (order of tensor dimensions: K, I, J)
	##==== sample U
	print "sample U..."
	## Y_reshape
	Y_reshape = np.zeros((I, K, J))
	for i in range(I):
		for k in range(K):
			for j in range(J):
				Y_reshape[(i,k,j)] = Y[(k,i,j)]
	## coef_tensor
	T_reshape = np.transpose(T)
	V_reshape = np.transpose(V)
	coef_tensor = []
	for d in range(D):
		array = np.outer(T_reshape[d], V_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for k in range(K):
		coef_tensor_reshape.append([])
		for j in range(J):
			array = np.multiply(T[k], V[j])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## TV_lamb_list
	TV_lamb_list = np.zeros(D)
	for d in range(D):
		TV_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for i in range(I):
		for d in range(D):
			x = U[(i,d)]					## good
			data = Y_reshape[i]				## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([U[i]])
			array = U[i]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = TV_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			U[(i,d)] = draw


	##==== sample V
	print "sample V..."
	## Y_reshape
	Y_reshape = np.zeros((J, K, I))
	for j in range(J):
		for k in range(K):
			for i in range(I):
				Y_reshape[(j,k,i)] = Y[(k,i,j)]
	## coef_tensor
	T_reshape = np.transpose(T)
	U_reshape = np.transpose(U)
	coef_tensor = []
	for d in range(D):
		array = np.outer(T_reshape[d], U_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for k in range(K):
		coef_tensor_reshape.append([])
		for i in range(I):
			array = np.multiply(T[k], U[i])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## TU_lamb_list
	TU_lamb_list = np.zeros(D)
	for d in range(D):
		TU_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for j in range(J):
		for d in range(D):
			x = V[(j,d)]					## good
			data = Y_reshape[j]				## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([V[j]])
			array = V[j]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = TU_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			V[(j,d)] = draw


	##==== sample T
	print "sample T..."
	## coef_tensor
	U_reshape = np.transpose(U)
	V_reshape = np.transpose(V)
	coef_tensor = []
	for d in range(D):
		array = np.outer(U_reshape[d], V_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for i in range(I):
		coef_tensor_reshape.append([])
		for j in range(J):
			array = np.multiply(U[i], V[j])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## UV_lamb_list
	UV_lamb_list = np.zeros(D)
	for d in range(D):
		UV_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for k in range(K):
		for d in range(D):
			x = T[(k,d)]					## good
			data = Y[k]						## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([T[k]])
			array = T[k]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = UV_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			T[(k,d)] = draw



	return






def sampler_precision(alpha0, beta0):			## (global function)
	global Y, U, V, T, I, J, K, alpha

	##==== pre-cal
	Y_exp = tensor_cal(T, U, V)

	##==== sample alpha
	print "sample alpha..."
	alpha_new = alpha0 + 0.5 * (I * J * K)
	temp = np.sum( np.power(Y - Y_exp, 2) )
	beta_new = beta0 + 0.5 * temp
	shape = alpha_new
	scale = 1.0 / beta_new
	draw = np.random.gamma(shape, scale)
	alpha = draw
	return






##=================
## loglike
##=================
def loglike_Gaussian(obs, mu, sigma):			## NOTE: removed constant terms
	loglike = 0
	#loglike += - np.log( sigma * np.sqrt(2 * np.pi) )
	loglike += ( - np.log( sigma ) )
	loglike += ( - (obs - mu)**2 / (2 * sigma**2) )
	return loglike


## loglike of Gaussian's for a matrix with given mean value
def loglike_gaussian_uni(data, mu, lamb):		## NOTE: removed constant terms
	sigma = math.sqrt(1.0 / lamb)

	loglike = 0
	amount = len(data) * len(data[0])
	loglike += amount * ( - np.log( sigma ) )
	loglike += (- np.sum( np.power(data - mu, 2) / (2 * sigma**2) ) )
	return loglike


## loglike of Gaussian's for a tensor with given mean tensor
def loglike_gaussian_tensor(data, Mu, lamb):
	sigma = math.sqrt(1.0 / lamb)

	loglike = 0
	amount = len(data) * len(data[0]) * len(data[0][0])
	loglike += amount * ( - np.log( sigma ) )
	loglike += (- np.sum( np.power(data - Mu, 2) / (2 * sigma**2) ) )
	return loglike



## loglike og Gamma
def loglike_gamma(obs, alpha, beta):
	loglike = 0

	loglike += (alpha - 1) * np.log(obs)
	loglike += (- beta * obs)
	loglike += alpha * np.log(beta)
	loglike += (- np.log( sps.gamma(alpha) ) )
	return loglike



def loglike_cal(mu0, lamb0, alpha0, beta0):							## (global function)
	global Y, U, V, T, alpha, K, I, J, D
	global loglike_total, loglike_Y, loglike_U, loglike_V, loglike_T, loglike_alpha

	print "calculating the loglike..."
	loglike_cumu = 0

	##==== loglike_Y
	mean = tensor_cal(T, U, V)
	loglike = loglike_gaussian_tensor(Y, mean, alpha)
	loglike_Y.append(loglike)
	np.save("./result/loglike_Y", np.array(loglike_Y))
	loglike_cumu += loglike
	print "loglike Y:", loglike


	##==== loglike_U
	loglike = loglike_gaussian_uni(U, mu0, lamb0)
	loglike_U.append(loglike)
	np.save("./result/loglike_U", np.array(loglike_U))
	loglike_cumu += loglike
	print "loglike U:", loglike


	##==== loglike_V
	loglike = loglike_gaussian_uni(V, mu0, lamb0)
	loglike_V.append(loglike)
	np.save("./result/loglike_V", np.array(loglike_V))
	loglike_cumu += loglike
	print "loglike V:", loglike


	##==== loglike_T
	loglike = loglike_gaussian_uni(T, mu0, lamb0)
	loglike_T.append(loglike)
	np.save("./result/loglike_T", np.array(loglike_T))
	loglike_cumu += loglike
	print "loglike T:", loglike


	##==== loglike_alpha
	loglike = loglike_gamma(alpha, alpha0, beta0)
	loglike_alpha.append(loglike)
	np.save("./result/loglike_alpha", np.array(loglike_alpha))
	loglike_cumu += loglike
	print "loglike alpha:", loglike


	##==== loglike_total
	loglike_total.append(loglike_cumu)
	np.save("./result/loglike_total", np.array(loglike_total))
	print "loglike total:", loglike_cumu


	return







##=================
## main
##=================
if __name__ == '__main__':



	##==== load parameters
	## data
	Y = np.load("./data_simu/Y.npy")
	## para init
	U = np.load("./data_init/U.npy")
	V = np.load("./data_init/V.npy")
	T = np.load("./data_init/T.npy")
	alpha = np.load("./data_init/alpha.npy")[0]



	##==== fill in the dimensions
	shape = Y.shape
	K = shape[0]
	I = shape[1]
	J = shape[2]
	shape = U.shape
	D = shape[1]


	##==== simulation
	## I will treat all observed variables as input of simu functions
	mu = 0						## TODO
	lamb = 1.0					## TODO
	alpha0 = 1.0				## TODO
	beta0 = 0.5					## TODO


	loglike_total = []
	loglike_Y = []
	loglike_U = []
	loglike_V = []
	loglike_T = []
	loglike_alpha = []


	ITER = 5000					## TODO
	for i in range(ITER):
		print "now working on iter#", i+1


		##==== timer
		start_time = timeit.default_timer()


		##==== sample all
		sampler_factor(alpha, mu, lamb)
		sampler_precision(alpha0, beta0)

		##==== cal loglike
		loglike_cal(mu, lamb, alpha0, beta0)


		##==== save parameters
		np.save("./result/U", U)
		np.save("./result/V", V)
		np.save("./result/T", T)
		np.save("./result/alpha", np.array([alpha]))



		##==== timer
		elapsed = timeit.default_timer() - start_time
		print "time per iteration:", elapsed




	## save learned parameters
	print "now saving the learned models..."
	np.save("./result/U", U)
	np.save("./result/V", V)
	np.save("./result/T", T)
	np.save("./result/alpha", np.array([alpha]))






