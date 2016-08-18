## this samples the data from simulator1.py: a symmetric tensor, whose factors are all with uniform Gaussian's

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
Y1 = []
U1 = []
V1 = []
T1 = []
Y2 = []
U2 = []
V2 = []
T2 = []
alpha = 1
# NOTE: the following are the real scenario for chr22 and brain tensor
K = 13
I = 159
J = 585
D1 = 40
D2 = 40


loglike_total = []
loglike_data = []
loglike_Y1 = []
loglike_Y2 = []
loglike_U1 = []
loglike_V1 = []
loglike_T1 = []
loglike_U2 = []
loglike_V2 = []
loglike_T2 = []
loglike_alpha = []






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



##=================
## sampler
##=================
## NOTE: here we assume no entries are missing in the tensor
def sampler_subT(lamb):								## (global function)
	global Y, Y1, Y2, U1, V1, T1, U2, V2, T2, I, J, K, alpha

	##==== sample Y1
	print "sample Y1..."
	#== pre-cal
	mean1 = tensor_cal(T1, U1, V1)
	mean2 = Y - Y2
	#== combine two Gaussian's
	lamb1 = lamb
	lamb2 = alpha
	lamb_new = lamb1 + lamb2
	mean_new = (lamb1 / lamb_new) * mean1 + (lamb2 / lamb_new) * mean2
	#== sampling
	sigma = math.sqrt(1.0 / lamb_new)
	for k in range(K):
		for i in range(I):
			for j in range(J):
				mu = mean_new[(k,i,j)]
				draw = np.random.normal(mu, sigma)
				Y1[(k,i,j)] = draw


	##==== sample Y2
	print "sample Y2..."
	#== pre-cal
	mean1 = tensor_cal(T2, U2, V2)
	mean2 = Y - Y1
	#== combine two Gaussian's
	lamb1 = lamb
	lamb2 = alpha
	lamb_new = lamb1 + lamb2
	mean_new = (lamb1 / lamb_new) * mean1 + (lamb2 / lamb_new) * mean2
	#== sampling
	sigma = math.sqrt(1.0 / lamb_new)
	for k in range(K):
		for i in range(I):
			for j in range(J):
				mu = mean_new[(k,i,j)]
				draw = np.random.normal(mu, sigma)
				Y2[(k,i,j)] = draw
	return


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



def sampler_factor(lamb0, mu1, lamb1):						## (global function)
	global Y, Y1, U1, V1, T1, Y2, U2, V2, T2, alpha, K, I, J, D1, D2


	##==== sample the others (order of tensor dimensions: K, I, J)
	##== Y1 relevant
	##==== sample U1
	print "sample U1..."
	## Y1_reshape
	Y1_reshape = np.zeros((I, K, J))
	for i in range(I):
		for k in range(K):
			for j in range(J):
				Y1_reshape[(i,k,j)] = Y1[(k,i,j)]
	## coef_tensor
	T1_reshape = np.transpose(T1)
	V1_reshape = np.transpose(V1)
	coef_tensor = []
	for d in range(D1):
		array = np.outer(T1_reshape[d], V1_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for k in range(K):
		coef_tensor_reshape.append([])
		for j in range(J):
			array = np.multiply(T1[k], V1[j])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## TV_lamb_list
	TV_lamb_list = np.zeros(D1)
	for d in range(D1):
		TV_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for i in range(I):
		for d in range(D1):
			x = U1[(i,d)]					## good
			data = Y1_reshape[i]			## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([U1[i]])
			array = U1[i]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = TV_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			U1[(i,d)] = draw

	##==== sample V1
	print "sample V1..."
	## Y1_reshape
	Y1_reshape = np.zeros((J, K, I))
	for j in range(J):
		for k in range(K):
			for i in range(I):
				Y1_reshape[(j,k,i)] = Y1[(k,i,j)]
	## coef_tensor
	T1_reshape = np.transpose(T1)
	U1_reshape = np.transpose(U1)
	coef_tensor = []
	for d in range(D1):
		array = np.outer(T1_reshape[d], U1_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for k in range(K):
		coef_tensor_reshape.append([])
		for i in range(I):
			array = np.multiply(T1[k], U1[i])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## TU_lamb_list
	TU_lamb_list = np.zeros(D1)
	for d in range(D1):
		TU_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for j in range(J):
		for d in range(D1):
			x = V1[(j,d)]					## good
			data = Y1_reshape[j]			## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([V1[j]])
			array = V1[j]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = TU_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			V1[(j,d)] = draw

	##==== sample T1
	print "sample T1..."
	## coef_tensor
	U1_reshape = np.transpose(U1)
	V1_reshape = np.transpose(V1)
	coef_tensor = []
	for d in range(D1):
		array = np.outer(U1_reshape[d], V1_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for i in range(I):
		coef_tensor_reshape.append([])
		for j in range(J):
			array = np.multiply(U1[i], V1[j])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## UV_lamb_list
	UV_lamb_list = np.zeros(D1)
	for d in range(D1):
		UV_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for k in range(K):
		for d in range(D1):
			x = T1[(k,d)]					## good
			data = Y1[k]					## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([T1[k]])
			array = T1[k]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = UV_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			T1[(k,d)] = draw




	##==== sample the others (order of tensor dimensions: K, I, J)
	##== Y2 relevant
	##==== sample U2
	print "sample U2..."
	## Y2_reshape
	Y2_reshape = np.zeros((I, K, J))
	for i in range(I):
		for k in range(K):
			for j in range(J):
				Y2_reshape[(i,k,j)] = Y2[(k,i,j)]
	## coef_tensor
	T2_reshape = np.transpose(T2)
	V2_reshape = np.transpose(V2)
	coef_tensor = []
	for d in range(D2):
		array = np.outer(T2_reshape[d], V2_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for k in range(K):
		coef_tensor_reshape.append([])
		for j in range(J):
			array = np.multiply(T2[k], V2[j])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## TV_lamb_list
	TV_lamb_list = np.zeros(D2)
	for d in range(D2):
		TV_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for i in range(I):
		for d in range(D2):
			x = U2[(i,d)]					## good
			data = Y2_reshape[i]			## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([U2[i]])
			array = U2[i]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = TV_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			U2[(i,d)] = draw

	##==== sample V2
	print "sample V2..."
	## Y2_reshape
	Y2_reshape = np.zeros((J, K, I))
	for j in range(J):
		for k in range(K):
			for i in range(I):
				Y2_reshape[(j,k,i)] = Y2[(k,i,j)]
	## coef_tensor
	T2_reshape = np.transpose(T2)
	U2_reshape = np.transpose(U2)
	coef_tensor = []
	for d in range(D2):
		array = np.outer(T2_reshape[d], U2_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for k in range(K):
		coef_tensor_reshape.append([])
		for i in range(I):
			array = np.multiply(T2[k], U2[i])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## TU_lamb_list
	TU_lamb_list = np.zeros(D2)
	for d in range(D2):
		TU_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for j in range(J):
		for d in range(D2):
			x = V2[(j,d)]					## good
			data = Y2_reshape[j]			## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([V2[j]])
			array = V2[j]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = TU_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			V2[(j,d)] = draw

	##==== sample T2
	print "sample T2..."
	## coef_tensor
	U2_reshape = np.transpose(U2)
	V2_reshape = np.transpose(V2)
	coef_tensor = []
	for d in range(D2):
		array = np.outer(U2_reshape[d], V2_reshape[d])
		coef_tensor.append(array)
	## coef_tensor_reshape
	coef_tensor_reshape = []
	for i in range(I):
		coef_tensor_reshape.append([])
		for j in range(J):
			array = np.multiply(U2[i], V2[j])
			coef_tensor_reshape[-1].append(array)
	coef_tensor_reshape = np.array(coef_tensor_reshape)
	## UV_lamb_list
	UV_lamb_list = np.zeros(D2)
	for d in range(D2):
		UV_lamb_list[d] = lamb0 * np.sum( np.power(coef_tensor[d], 2) ) + lamb1
	#### sampling
	for k in range(K):
		for d in range(D2):
			x = T2[(k,d)]					## good
			data = Y2[k]					## good

			#mean1 = Y_reshape_exp[i]		## xxx
			# re-calculate mean1
			#array = np.array([T2[k]])
			array = T2[k]
			mean1 = np.tensordot( array, coef_tensor_reshape, axes=([0,2]) )

			coef = coef_tensor[d]			## good

			lamb = UV_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			T2[(k,d)] = draw

	return






def sampler_precision(alpha0, beta0):			## (global function)
	global Y, Y1, Y2, I, J, K, alpha

	##==== sample alpha
	print "sample alpha..."
	alpha_new = alpha0 + 0.5 * (I * J * K)
	temp = np.sum( np.power(Y - Y1 - Y2, 2) )
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
	global Y, Y1, U1, V1, T1, Y2, U2, V2, T2, alpha, K, I, J, D1, D2
	global loglike_total, loglike_data, loglike_Y1, loglike_Y2
	global loglike_U1, loglike_V1, loglike_T1, loglike_U2, loglike_V2, loglike_T2, loglike_alpha

	print "calculating the loglike..."
	loglike_cumu = 0

	##==== loglike_data
	loglike = loglike_gaussian_tensor(Y, Y1 + Y2, alpha)
	loglike_data.append(loglike)
	np.save("./result/loglike_data", np.array(loglike_data))
	loglike_cumu += loglike
	print "loglike data:", loglike

	##==== loglike_Y1
	mean = tensor_cal(T1, U1, V1)
	loglike = loglike_gaussian_tensor(Y1, mean, lamb0)
	loglike_Y1.append(loglike)
	np.save("./result/loglike_Y1", np.array(loglike_Y1))
	loglike_cumu += loglike
	print "loglike Y1:", loglike

	##==== loglike_Y2
	mean = tensor_cal(T2, U2, V2)
	loglike = loglike_gaussian_tensor(Y2, mean, lamb0)
	loglike_Y2.append(loglike)
	np.save("./result/loglike_Y2", np.array(loglike_Y2))
	loglike_cumu += loglike
	print "loglike Y2:", loglike

	##==== loglike_U1
	loglike = loglike_gaussian_uni(U1, mu0, lamb0)
	loglike_U1.append(loglike)
	np.save("./result/loglike_U1", np.array(loglike_U1))
	loglike_cumu += loglike
	print "loglike U1:", loglike

	##==== loglike_V1
	loglike = loglike_gaussian_uni(V1, mu0, lamb0)
	loglike_V1.append(loglike)
	np.save("./result/loglike_V1", np.array(loglike_V1))
	loglike_cumu += loglike
	print "loglike V1:", loglike

	##==== loglike_T1
	loglike = loglike_gaussian_uni(T1, mu0, lamb0)
	loglike_T1.append(loglike)
	np.save("./result/loglike_T1", np.array(loglike_T1))
	loglike_cumu += loglike
	print "loglike T1:", loglike

	##==== loglike_U2
	loglike = loglike_gaussian_uni(U2, mu0, lamb0)
	loglike_U2.append(loglike)
	np.save("./result/loglike_U2", np.array(loglike_U2))
	loglike_cumu += loglike
	print "loglike U2:", loglike

	##==== loglike_V2
	loglike = loglike_gaussian_uni(V2, mu0, lamb0)
	loglike_V2.append(loglike)
	np.save("./result/loglike_V2", np.array(loglike_V2))
	loglike_cumu += loglike
	print "loglike V2:", loglike

	##==== loglike_T2
	loglike = loglike_gaussian_uni(T2, mu0, lamb0)
	loglike_T2.append(loglike)
	np.save("./result/loglike_T2", np.array(loglike_T2))
	loglike_cumu += loglike
	print "loglike T2:", loglike

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
	Y1 = np.load("./data_init/Y1.npy")
	U1 = np.load("./data_init/U1.npy")
	V1 = np.load("./data_init/V1.npy")
	T1 = np.load("./data_init/T1.npy")
	Y2 = np.load("./data_init/Y2.npy")
	U2 = np.load("./data_init/U2.npy")
	V2 = np.load("./data_init/V2.npy")
	T2 = np.load("./data_init/T2.npy")
	alpha = np.load("./data_init/alpha.npy")[0]



	##==== fill in the dimensions
	shape = Y.shape
	K = shape[0]
	I = shape[1]
	J = shape[2]
	shape = U1.shape
	D1 = shape[1]
	shape = U2.shape
	D2 = shape[1]



	##==== simulation
	## I will treat all observed variables as input of simu functions
	mu = 0						## TODO
	lamb = 1.0					## TODO
	alpha0 = 1.0				## TODO
	beta0 = 0.5					## TODO


	loglike_total = []
	loglike_data = []
	loglike_Y1 = []
	loglike_Y2 = []
	loglike_U1 = []
	loglike_V1 = []
	loglike_T1 = []
	loglike_U2 = []
	loglike_V2 = []
	loglike_T2 = []
	loglike_alpha = []


	ITER = 5000					## TODO
	for i in range(ITER):
		print "now working on iter#", i+1


		##==== timer
		start_time = timeit.default_timer()


		##==== sample all
		sampler_subT(lamb)
		sampler_factor(lamb, mu, lamb)
		sampler_precision(alpha0, beta0)

		##==== cal loglike
		loglike_cal(mu, lamb, alpha0, beta0)



		np.save("./result/Y1", Y1)
		np.save("./result/U1", U1)
		np.save("./result/V1", V1)
		np.save("./result/T1", T1)
		np.save("./result/Y2", Y2)
		np.save("./result/U2", U2)
		np.save("./result/V2", V2)
		np.save("./result/T2", T2)
		np.save("./result/alpha", np.array([alpha]))




		##==== timer
		elapsed = timeit.default_timer() - start_time
		print "time per iteration:", elapsed





	## save learned parameters
	print "now saving the learned models..."
	np.save("./result/Y1", Y1)
	np.save("./result/U1", U1)
	np.save("./result/V1", V1)
	np.save("./result/T1", T1)
	np.save("./result/Y2", Y2)
	np.save("./result/U2", U2)
	np.save("./result/V2", V2)
	np.save("./result/T2", T2)
	np.save("./result/alpha", np.array([alpha]))




