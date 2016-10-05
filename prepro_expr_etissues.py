## pick up etissues

## input:
##	1. "phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_tissue_type_count_0"
## output:
##	2. "Tissue_list.npy"




import numpy as np




## TODO: set threshold
count_etissue = 100




if __name__ == "__main__":


	##== extract
	list_etissues = []
	file = open("./data_raw/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt_tissue_type_count_0", 'r')
	while 1:
		line = (file.readline()).strip()
		if not line:
			break

		line = line.split('\t')
		tissue = line[0]
		count = int(line[1])
		if count >= count_etissue:
			list_etissues.append(tissue)
	file.close()
	list_etissues = np.array(list_etissues)
	print "there are",
	print len(list_etissues),
	print "etissues..."

	##== save
	np.save("./data_prepared/Tissue_list", list_etissues)




