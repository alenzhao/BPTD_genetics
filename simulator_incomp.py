## this is used to make the expression tensor incomplete


## input: ./data_simu/Y.npy
## output: ./data_simu_init/Tensor_tissue_k.txt



import numpy as np




if __name__ == "__main__":



	print "making the tensor incomplete..."



	##========================================================================================
	##======== load full tensor
	##========================================================================================
	Y = np.load("./data_simu/Y.npy")
	shape = Y.shape
	K = shape[0]
	I = shape[1]
	J = shape[2]
	print "full tensor dimension:", K, I, J


	##========================================================================================
	##======== make it incomplete, and save
	##========================================================================================
	repo_temp = {}
	for k in range(K):
		print "working on tissue#",
		print k

		## random draw
		Y_sub = []
		list_pos = []
		list_sample = []
		arr_permute = np.random.permutation(I)
		topper = int(I*0.5)							## NOTE: to choose the portion of incomplete tensor
		for pos in arr_permute[:topper]:
			#
			repo_temp[pos] = 1
			#
			list_pos.append(pos)
			Y_sub.append(Y[k][pos])
			sample = str(k) + "-" + str(pos)
			list_sample.append(sample)
		list_pos = np.array(list_pos)
		list_sample = np.array(list_sample)
		Y_sub = np.array(Y_sub)
		np.save("./data_simu_init/Tensor_tissue_" + str(k) + "_list", list_sample)
		np.save("./data_simu_init/Tensor_tissue_" + str(k), Y_sub)

		## save
		file = open("./data_simu_init/Tensor_tissue_" + str(k) + ".txt", 'w')
		for i in range(len(list_pos)):
			pos = list_pos[i]
			file.write(str(pos) + '\t')
			for y in Y_sub[i]:
				file.write(str(y) + '\t')
			file.write('\n')
		file.close()


	##==== test
	print "test: num of individuals and size of picked repo are",
	print I,
	print len(repo_temp)







