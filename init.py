import numpy as np



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




if __name__=="__main__":



	##==== X (genotype)
	list_individual = np.load("./data_real/tensor/Individual_list.npy")
	X = []
	for individual in list_individual:
		X.append([])
		file = open("./data_real/genotype_450_dosage_matrix_qc/chr22/SNP_dosage_" + individual + ".txt", 'r')
		while 1:
			line = (file.readline()).strip()
			if not line:
				break

			X[-1].append(float(line))
		file.close()
	X = np.array(X)
	print "X shape:", X.shape
	np.save("./data_real/X", X)




	##==== init model
	U = np.load("./data_real/tensor/Individual.npy")
	V = np.load("./data_real/tensor/Gene.npy")
	T = np.load("./data_real/tensor/Tissue.npy")

	np.save("./data_real/U1", U)
	np.save("./data_real/V1", V)
	np.save("./data_real/T1", 0.5 * T)

	np.save("./data_real/U2", U)
	np.save("./data_real/V2", V)
	np.save("./data_real/T2", 0.5 * T)

	tensor = 0.5 * tensor_cal(T, U, V)
	np.save("./data_real/Y1", tensor)
	np.save("./data_real/Y2", tensor)

	# U + X -> Beta: X * Beta_inv = U
	model = np.linalg.lstsq(X, U)
	Beta = np.transpose(model[0])
	np.save("./data_real/Beta", Beta)





	print "prepare done..."







