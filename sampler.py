## simulate from real genotype data and the model
## simulation is for testing purpose (convergence)



# TODO: didn't implement the incomplete tensor



##==== libraries
import math
import numpy as np
import scipy.special as sps



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
loglike_Beta = []
loglike_alpha = []







##=================
## sampler
##=================
## NOTE: here we assume no entries are missing in the tensor
def sampler_subT(lamb):								## (global function)
	global Y, Y1, Y2, U1, V1, T1, U2, V2, T2, I, J, K, alpha

	##==== sample Y1
	print "sample Y1..."
	#== pre-cal
	temp_T = []
	for i in range(I):
		temp_T.append([])
		for j in range(J):
			temp_array = np.multiply(U1[i], V1[j])
			temp_T[-1].append(temp_array)
	mean1 = []
	for k in range(K):
		mean1.append([])
		for i in range(I):
			mean1[-1].append([])
			for j in range(J):
				value = np.inner(temp_T[i][j], T1[k])
				mean1[-1][-1].append(value)
	mean1 = np.array(mean1)
	mean2 = Y - Y2

	#== combine two Gaussian's
	lamb1 = lamb
	lamb2 = alpha
	lamb_new = lamb1 + lamb2
	mean_new = (lamb1 / lamb_new) * mean1 + (lamb2 / lamb_new) * mean2

	#== sample
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
	temp_T = []
	for i in range(I):
		temp_T.append([])
		for j in range(J):
			temp_array = np.multiply(U2[i], V2[j])
			temp_T[-1].append(temp_array)
	mean1 = []
	for k in range(K):
		mean1.append([])
		for i in range(I):
			mean1[-1].append([])
			for j in range(J):
				value = np.inner(temp_T[i][j], T2[k])
				mean1[-1][-1].append(value)
	mean1 = np.array(mean1)
	mean2 = Y - Y1

	#== combine two Gaussian's
	lamb1 = lamb
	lamb2 = alpha
	lamb_new = lamb1 + lamb2
	mean_new = (lamb1 / lamb_new) * mean1 + (lamb2 / lamb_new) * mean2

	#== sample
	sigma = math.sqrt(1.0 / lamb_new)
	for k in range(K):
		for i in range(I):
			for j in range(J):
				mu = mean_new[(k,i,j)]
				draw = np.random.normal(mu, sigma)
				Y2[(k,i,j)] = draw
	return




def GaussianComb_2D(x, data, mean1, coef, lamb1, mean2, lamb2):
	## likelihood: x, data, mean1, coef, lamb1
	## prior: mean2, lamb2

	list_mean = []
	list_lamb = []
	dimension = len(data)
	coef_sq = np.power(coef, 2)
	for i in range(dimension):
		if coef[i] == 0:
			continue
		list_mean.append( (data[i] - mean1[i] + coef[i] * x) / coef[i] )
		list_lamb.append( coef_sq[i] )
	list_mean = np.array(list_mean)
	list_lamb = np.array(list_lamb)
	list_lamb = list_lamb * lamb1

	# we need to combine len(list_mean) + 1 Gaussian's
	dimension = len(list_mean)
	lamb_new = np.sum(list_lamb) + lamb2

	list_lamb_scale = list_lamb / lamb_new
	lamb2_scale = lamb2 / lamb_new
	mean_new = np.sum( np.multiply(list_lamb_scale, list_mean) )
	mean_new += lamb2_scale * mean2

	sigma = math.sqrt(1.0 / lamb_new)
	mu = mean_new
	draw = np.random.normal(mu, sigma)
	return draw




def GaussianComb_3D(x, data, mean1, coef, lamb1, mean2, lamb2):
	## likelihood: x, data, mean1, coef, lamb1
	## prior: mean2, lamb2

	list_mean = []
	list_lamb = []
	shape = data.shape
	dimension1 = shape[0]
	dimension2 = shape[1]
	coef_sq = np.power(coef, 2)
	for i in range(dimension1):
		for j in range(dimension2):
			if coef[i][j] == 0:
				continue
			list_mean.append( (data[i][j] - mean1[i][j] + coef[i][j] * x) / coef[i][j] )
			list_lamb.append( coef_sq[i][j] )
	list_mean = np.array(list_mean)
	list_lamb = np.array(list_lamb)
	list_lamb = list_lamb * lamb1

	# we need to combine len(list_mean) + 1 Gaussian's
	dimension = len(list_mean)
	lamb_new = np.sum(list_lamb) + lamb2

	list_lamb_scale = list_lamb / lamb_new
	lamb2_scale = lamb2 / lamb_new
	mean_new = np.sum( np.multiply(list_lamb_scale, list_mean) )
	mean_new += lamb2_scale * mean2

	sigma = math.sqrt(1.0 / lamb_new)
	mu = mean_new
	draw = np.random.normal(mu, sigma)
	return draw



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
	global X, Y, Y1, U1, V1, T1, Beta, Y2, U2, V2, T2, alpha, K, I, J, S, D1, D2

	##==== sample Beta
	print "sample Beta..."
	U1_reshape = np.transpose(U1)
	X_reshape = np.transpose(X)
	U1_reshape_exp = np.dot(Beta, X_reshape)
	Beta_reshape = np.transpose(Beta)
	for d in range(D1):
		for s in range(S):
			beta = Beta[(d, s)]
			data = U1_reshape[d]
			mean1 = U1_reshape_exp[d]
			coef = X_reshape[s]
			Beta[(d, s)] = GaussianComb_2D(beta, data, mean1, coef, lamb0, mu1, lamb1)


	##==== sample the others (order of tensor dimensions: K, I, J)
	##== Y1 relevant
	Y1_exp = tensor_cal(T1, U1, V1)

	#== sample U1
	print "sample U1..."
	Y1_reshape = np.zeros((I, K, J))
	for i in range(I):
		for k in range(K):
			for j in range(J):
				Y1_reshape[(i,k,j)] = Y1[(k,i,j)]
	Y1_reshape_exp = np.zeros((I, K, J))
	for i in range(I):
		for k in range(K):
			for j in range(J):
				Y1_reshape_exp[(i,k,j)] = Y1_exp[(k,i,j)]
	T_reshape = np.transpose(T1)
	V_reshape = np.transpose(V1)
	coef_tensor = []
	for d in range(D1):
		array = np.outer(T_reshape[d], V_reshape[d])
		coef_tensor.append(array)
	Beta_reshape = np.transpose(Beta)
	mu_matrix = np.dot(X, Beta_reshape)
	for i in range(I):
		for d in range(D1):
			u = U1[(i,d)]
			data = Y1_reshape[i]
			mean1 = Y1_reshape_exp[i]
			coef = coef_tensor[d]
			mu = mu_matrix[i][d]
			U1[(i,d)] = GaussianComb_3D(u, data, mean1, coef, lamb0, mu, lamb0)

	#== sample V1
	print "sample V1..."
	Y1_reshape = np.zeros((J, K, I))
	for j in range(J):
		for k in range(K):
			for i in range(I):
				Y1_reshape[(j,k,i)] = Y1[(k,i,j)]
	Y1_reshape_exp = np.zeros((J, K, I))
	for j in range(J):
		for k in range(K):
			for i in range(I):
				Y1_reshape_exp[(j,k,i)] = Y1_exp[(k,i,j)]
	T_reshape = np.transpose(T1)
	U_reshape = np.transpose(U1)
	coef_tensor = []
	for d in range(D1):
		array = np.outer(T_reshape[d], U_reshape[d])
		coef_tensor.append(array)
	for j in range(J):
		for d in range(D1):
			v = V1[(j,d)]
			data = Y1_reshape[j]
			mean1 = Y1_reshape_exp[j]
			coef = coef_tensor[d]
			V1[(j,d)] = GaussianComb_3D(v, data, mean1, coef, lamb0, mu1, lamb1)

	#== sample T1
	print "sample T1..."
	U_reshape = np.transpose(U1)
	V_reshape = np.transpose(V1)
	coef_tensor = []
	for d in range(D1):
		array = np.outer(U_reshape[d], V_reshape[d])
		coef_tensor.append(array)
	for k in range(K):
		for d in range(D1):
			t = T1[(k,d)]
			data = Y1[k]
			mean1 = Y1_exp[k]
			coef = coef_tensor[d]
			V1[(j,d)] = GaussianComb_3D(t, data, mean1, coef, lamb0, mu1, lamb1)




	##== Y2 relevant
	Y2_exp = tensor_cal(T2, U2, V2)

	#== sample U2
	print "sample U2..."
	Y2_reshape = np.zeros((I, K, J))
	for i in range(I):
		for k in range(K):
			for j in range(J):
				Y2_reshape[(i,k,j)] = Y2[(k,i,j)]
	Y2_reshape_exp = np.zeros((I, K, J))
	for i in range(I):
		for k in range(K):
			for j in range(J):
				Y2_reshape_exp[(i,k,j)] = Y2_exp[(k,i,j)]
	T_reshape = np.transpose(T2)
	V_reshape = np.transpose(V2)
	coef_tensor = []
	for d in range(D2):
		array = np.outer(T_reshape[d], V_reshape[d])
		coef_tensor.append(array)
	for i in range(I):
		for d in range(D2):
			u = U2[(i,d)]
			data = Y2_reshape[i]
			mean1 = Y2_reshape_exp[i]
			coef = coef_tensor[d]
			U2[(i,d)] = GaussianComb_3D(u, data, mean1, coef, lamb0, mu1, lamb1)

	#== sample V2
	print "sample V2..."
	Y2_reshape = np.zeros((J, K, I))
	for j in range(J):
		for k in range(K):
			for i in range(I):
				Y2_reshape[(j,k,i)] = Y2[(k,i,j)]
	Y2_reshape_exp = np.zeros((J, K, I))
	for j in range(J):
		for k in range(K):
			for i in range(I):
				Y2_reshape_exp[(j,k,i)] = Y2_exp[(k,i,j)]
	T_reshape = np.transpose(T2)
	U_reshape = np.transpose(U2)
	coef_tensor = []
	for d in range(D2):
		array = np.outer(T_reshape[d], U_reshape[d])
		coef_tensor.append(array)
	for j in range(J):
		for d in range(D2):
			v = V2[(j,d)]
			data = Y2_reshape[j]
			mean1 = Y2_reshape_exp[j]
			coef = coef_tensor[d]
			V2[(j,d)] = GaussianComb_3D(v, data, mean1, coef, lamb0, mu1, lamb1)

	#== sample T2
	print "sample T2..."
	U_reshape = np.transpose(U2)
	V_reshape = np.transpose(V2)
	coef_tensor = []
	for d in range(D2):
		array = np.outer(U_reshape[d], V_reshape[d])
		coef_tensor.append(array)
	for k in range(K):
		for d in range(D2):
			t = T2[(k,d)]
			data = Y2[k]
			mean1 = Y2_exp[k]
			coef = coef_tensor[d]
			T2[(k,d)] = GaussianComb_3D(t, data, mean1, coef, lamb0, mu1, lamb1)


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


## loglike of Gaussian's for a matrix with given mean matrix
def loglike_gaussian_Mu(data, Mu, lamb):
	sigma = math.sqrt(1.0 / lamb)

	loglike = 0
	amount = len(data) * len(data[0])
	loglike += amount * ( - np.log( sigma ) )
	loglike += (- np.sum( np.power(data - Mu, 2) / (2 * sigma**2) ) )
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
	global X, Y, Y1, U1, V1, T1, Beta, Y2, U2, V2, T2, alpha, K, I, J, S, D1, D2
	global loglike_total, loglike_data, loglike_Y1, loglike_Y2
	global loglike_U1, loglike_V1, loglike_T1, loglike_U2, loglike_V2, loglike_T2, loglike_Beta, loglike_alpha

	print "calculating the loglike..."
	loglike_cumu = 0

	##==== loglike_data
	loglike = loglike_gaussian_tensor(Y, Y1 + Y2, alpha)
	loglike_data.append(loglike)
	np.save("./result/loglike_data", np.array(loglike_data))
	loglike_cumu += loglike

	##==== loglike_Y1
	mean = tensor_cal(T1, U1, V1)
	loglike = loglike_gaussian_tensor(Y1, mean, lamb0)
	loglike_Y1.append(loglike)
	np.save("./result/loglike_Y1", np.array(loglike_Y1))
	loglike_cumu += loglike

	##==== loglike_Y2
	mean = tensor_cal(T2, U2, V2)
	loglike = loglike_gaussian_tensor(Y2, mean, lamb0)
	loglike_Y2.append(loglike)
	np.save("./result/loglike_Y2", np.array(loglike_Y2))
	loglike_cumu += loglike

	##==== loglike_U1
	Beta_reshape = np.transpose(Beta)
	mean = np.dot(X, Beta_reshape)
	loglike = loglike_gaussian_Mu(U1, mean, lamb0)
	loglike_U1.append(loglike)
	np.save("./result/loglike_U1", np.array(loglike_U1))
	loglike_cumu += loglike

	##==== loglike_V1
	loglike = loglike_gaussian_uni(V1, mu0, lamb0)
	loglike_V1.append(loglike)
	np.save("./result/loglike_V1", np.array(loglike_V1))
	loglike_cumu += loglike

	##==== loglike_T1
	loglike = loglike_gaussian_uni(T1, mu0, lamb0)
	loglike_T1.append(loglike)
	np.save("./result/loglike_T1", np.array(loglike_T1))
	loglike_cumu += loglike

	##==== loglike_U2
	loglike = loglike_gaussian_uni(U2, mu0, lamb0)
	loglike_U2.append(loglike)
	np.save("./result/loglike_U2", np.array(loglike_U2))
	loglike_cumu += loglike

	##==== loglike_V2
	loglike = loglike_gaussian_uni(V2, mu0, lamb0)
	loglike_V2.append(loglike)
	np.save("./result/loglike_V2", np.array(loglike_V2))
	loglike_cumu += loglike

	##==== loglike_T2
	loglike = loglike_gaussian_uni(T2, mu0, lamb0)
	loglike_T2.append(loglike)
	np.save("./result/loglike_T2", np.array(loglike_T2))
	loglike_cumu += loglike

	##==== loglike_Beta
	loglike = loglike_gaussian_uni(Beta, mu0, lamb0)
	loglike_Beta.append(loglike)
	np.save("./result/loglike_Beta", np.array(loglike_Beta))
	loglike_cumu += loglike

	##==== loglike_alpha
	loglike = loglike_gamma(alpha, alpha0, beta0)
	loglike_alpha.append(loglike)
	np.save("./result/loglike_alpha", np.array(loglike_alpha))
	loglike_cumu += loglike

	##==== loglike_total
	loglike_total.append(loglike_cumu)
	np.save("./result/loglike_total", np.array(loglike_total))

	return







##=================
## main
##=================
if __name__ == '__main__':



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



	##==== load parameters
	Y = np.load("./data_simu/Y.npy")
	Y1 = np.load("./data_simu/Y1.npy")
	U1 = np.load("./data_simu/U1.npy")
	Beta = np.load("./data_simu/Beta.npy")
	V1 = np.load("./data_simu/V1.npy")
	T1 = np.load("./data_simu/T1.npy")
	Y2 = np.load("./data_simu/Y2.npy")
	U2 = np.load("./data_simu/U2.npy")
	V2 = np.load("./data_simu/V2.npy")
	T2 = np.load("./data_simu/T2.npy")
	alpha = np.load("./data_simu/alpha.npy")[0]



	##==== fill in the dimensions
	shape = Y.shape
	K = shape[0]
	I = shape[1]
	J = shape[2]
	shape = Beta.shape
	S = shape[1]
	D1 = shape[0]
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
	loglike_Beta = []
	loglike_alpha = []


	ITER = 500					## TODO
	for i in range(ITER):
		print "now working on iter#", i

		##==== sample all
		sampler_subT(lamb)
		sampler_factor(lamb, mu, lamb)
		sampler_precision(alpha0, beta0)

		##==== cal loglike
		loglike_cal(mu, lamb, alpha0, beta0)


	## save learned parameters
	print "now saving the learned models..."

	np.save("./result/Y1", Y1)
	np.save("./result/U1", U1)
	np.save("./result/Beta", Beta)
	np.save("./result/V1", V1)
	np.save("./result/T1", T1)
	np.save("./result/Y2", Y2)
	np.save("./result/U2", U2)
	np.save("./result/V2", V2)
	np.save("./result/T2", T2)
	np.save("./result/alpha", np.array([alpha]))



