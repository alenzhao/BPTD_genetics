import matplotlib.pyplot as plt
import numpy as np



if __name__=="__main__":



	##==== total likelihood
	#arr = np.load("result/loglike_total.npy")


	#arr = [-1230398.625000, -1183251.000000, -1121522.375000, -1068431.875000, -1027335.250000, -987793.250000, -958500.937500, -925794.437500, -900950.875000, -872542.937500]



	arr = []
	file = open("./result/loglike_total_online.txt", 'r')
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		arr.append(float(line))
	file.close()





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
	plt.plot(arr[1:], 'r')





	plt.xlabel("Number of Iterations")
	plt.ylabel("Joint Log Likelihood")
	plt.title("Joint Log Likelihood during Model Training")
	plt.show()




