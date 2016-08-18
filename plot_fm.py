import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv



if __name__== "__main__":


	fm_loading = np.load('data_simu/V.npy')						# TODO

#	fm_loading = np.load('data_init/Lambda_gene.npy')					# TODO
#	fm_loading = np.load('result/fm_tissue.npy')						# TODO
#	fm_loading = inv(fm_loading)

#	cov = np.cov(fm_loading, rowvar=0)
#	fm_loading = cov

#	precision = np.array(inv(cov))
#	fm_loading = precision



	sns.set(context="paper", font="monospace")
	f, ax = plt.subplots(figsize=(22, 19))	# TODO
#	f, ax = plt.subplots(figsize=(26, 9))
	

	#sns_plot = sns.heatmap(fm_loading, xticklabels=x_label, yticklabels=y_label)
	sns_plot = sns.heatmap(fm_loading)
	#sns_plot = sns.heatmap(fm_loading, yticklabels=y_label)
	ax.set_xlabel('Factors')
	ax.set_ylabel('Individuals')			# TODO
#	plt.yticks(rotation=0)
#	plt.show()

	fig = sns_plot.get_figure()
	#fig.savefig("plot/quantile_c22_gene.jpg")
	#fig.savefig("/Users/shuoyang/Desktop/fm_gene.jpg")
	fig.savefig("/Users/shuoyang/Desktop/fm_heatmap.jpg")


	print "done..."



