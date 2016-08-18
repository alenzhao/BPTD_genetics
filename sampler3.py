## this samples the data from simulator3.py: a matrix

## simulate from real genotype data and the model
## simulation is for testing purpose (convergence)

# TODO: didn't implement the incomplete tensor



##==== libraries
import math
import numpy as np
import scipy.special as sps



##==== global variables
Y = []
U = []
V = []
alpha = 1
# NOTE: the following are the real scenario for chr22 and brain tensor
I = 159
J = 585
D = 40


loglike_total = []
loglike_Y = []
loglike_U = []
loglike_V = []
loglike_alpha = []





##=================
## sampler
##=================
# will do the following in the main function
"""
def LaplaceApprox_2D(x, data, mean1, coef, lamb1, mean2, lamb2):
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
	global Y, U, V, alpha, I, J, D

	##==== sample the others (order of tensor dimensions: K, I, J)
	V_reshape = np.transpose(V)
	Y_exp = np.dot(U, V_reshape)


	#== sample U
	print "sample U..."
	V_reshape = np.transpose(V)
	## lamb can be pre-cal
	V_lamb_list = np.zeros(len(V_reshape))
	for d in range(len(V_lamb_list)):
		V_lamb_list[d] = lamb0 * np.sum( np.power(V_reshape[d], 2) ) + lamb1
	for i in range(I):
		for d in range(D):
			x = U[(i,d)]			# good
			data = Y[i]				# good

			#mean1 = Y_exp[i]		# xxx
			# re-calculate mean1
			array = np.array([U[i]])
			mean1 = np.dot(array, V_reshape)[0]

			coef = V_reshape[d]		# good

			lamb = V_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			U[(i,d)] = draw



	#== sample V
	print "sample V..."
	Y_reshape = np.transpose(Y)
	Y_reshape_exp = np.transpose(Y_exp)
	U_reshape = np.transpose(U)
	## lamb can be pre-cal
	U_lamb_list = np.zeros(len(U_reshape))
	for d in range(len(U_lamb_list)):
		U_lamb_list[d] = lamb0 * np.sum( np.power(U_reshape[d], 2) ) + lamb1
	for j in range(J):
		for d in range(D):
			x = V[(j,d)]				# good
			data = Y_reshape[j]			# good

			#mean1 = Y_reshape_exp[j]	# xxx
			# re-calculate mean1
			array = np.array([V[j]])
			mean1 = np.dot(array, U_reshape)[0]

			coef = U_reshape[d]

			lamb = U_lamb_list[d]
			mean = lamb0 * np.sum( np.multiply( coef, (data - mean1 + coef * x) ) ) + lamb1 * mu1
			mean = mean / lamb

			##==== sampler
			sigma = math.sqrt(1.0 / lamb)
			mu = mean
			draw = np.random.normal(mu, sigma)
			V[(j,d)] = draw


	return






def sampler_precision(alpha0, beta0):			## (global function)
	global Y, U, V, I, J, alpha

	##==== pre-cal
	V_reshape = np.transpose(V)
	Y_exp = np.dot(U, V_reshape)

	##==== sample alpha
	print "sample alpha..."
	alpha_new = alpha0 + 0.5 * (I * J)
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


## loglike of Gaussian's for a matrix with given mean matrix
def loglike_gaussian_Mu(data, Mu, lamb):
	sigma = math.sqrt(1.0 / lamb)

	loglike = 0
	amount = len(data) * len(data[0])
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
	global Y, U, V, alpha, I, J, D
	global loglike_total, loglike_Y, loglike_U, loglike_V, loglike_alpha

	print "calculating the loglike..."
	loglike_cumu = 0

	##==== loglike_Y
	V_reshape = np.transpose(V)
	mean = np.dot(U, V_reshape)
	loglike = loglike_gaussian_Mu(Y, mean, alpha)
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
	Y = np.load("./data_simu/Y.npy")
	U = np.load("./data_simu/U.npy")
	V = np.load("./data_simu/V.npy")
	alpha = np.load("./data_simu/alpha.npy")[0]



	##==== fill in the dimensions
	shape = Y.shape
	I = shape[0]
	J = shape[1]
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
	loglike_alpha = []


	ITER = 5000					## TODO
	for i in range(ITER):
		print "now working on iter#", i+1

		##==== sample all
		sampler_factor(alpha, mu, lamb)
		sampler_precision(alpha0, beta0)

		##==== cal loglike
		loglike_cal(mu, lamb, alpha0, beta0)


		##==== save parameters
		np.save("./result/U", U)
		np.save("./result/V", V)
		np.save("./result/alpha", np.array([alpha]))




	## save learned parameters
	print "now saving the learned models..."
	np.save("./result/U", U)
	np.save("./result/V", V)
	np.save("./result/alpha", np.array([alpha]))




