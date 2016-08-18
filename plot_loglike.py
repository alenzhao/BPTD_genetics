import matplotlib.pyplot as plt
import numpy as np



if __name__=="__main__":



	##==== total likelihood
	arr = np.load("result/loglike_total.npy")


	##==== other likelihood terms
	#arr = np.load("result/loglike_data.npy")
	#arr = np.load("result/loglike_Y1.npy")
	#arr = np.load("result/loglike_U1.npy")
	#arr = np.load("result/loglike_V1.npy")
	#arr = np.load("result/loglike_T1.npy")
	#arr = np.load("result/loglike_Y2.npy")
	#arr = np.load("result/loglike_U2.npy")
	#arr = np.load("result/loglike_V2.npy")
	#arr = np.load("result/loglike_T2.npy")
	#arr = np.load("result/loglike_Beta.npy")
	#arr = np.load("result/loglike_alpha.npy")


	#arr = np.load("result/loglike_Y.npy")
	#arr = np.load("result/loglike_U.npy")
	#arr = np.load("result/loglike_V.npy")
	#arr = np.load("result/loglike_T.npy")
	#arr = np.load("result/loglike_alpha.npy")


	print arr
	plt.plot(arr, 'r')





	plt.xlabel("Number of Iterations")
	plt.ylabel("Joint Log Likelihood")
	plt.title("Joint Log Likelihood during Model Training")
	plt.show()




