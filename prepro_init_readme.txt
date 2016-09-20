The layout of the processing pipeline:

# what do we have (in running order):
	1.1. prepro_expr_rmnullgeno.py
	1.2. prepro_expr_etissues.py
	1.3. prepro_expr_etissue_norm.py
	2.1. init_1_Tensor.py (init_1_Tensor_incompPCA.R)
	2.2. init_2_Beta.py
	2.3. init_3_npy_to_txt.py

# what do they do:
	1.1. remove samples missing genotype information
	1.2. extract etissue list
	1.3. samples in etissues, rm null genes, and normalize
	2.1. init the two factor tensors
	2.2. reformat SNP data (with prepared dosage data), and init SNP-factor Beta
	2.3. reformat data from .npy to .txt (some)

# where are they:
	"project folder/data processing/"
	project folder: "GTEx_tensor_genetics"
	data processing: "preprocess_wholegenome"

# extra notes:
	1. gene expression data doesn't need to be preprocessed too much
	2. dosage data needs to be prepared (probably parsing from vcf files --> we have that script and pipeline before)

# other layout: (in cluster)
	1. "project folder/data processing/data_raw"				(all raw data)
	2. "project folder/data processing/data_prepared"			(used for save the prepared but un-normalized tensor data -- gene expression)
	3. "project folder/data processing/data_inter"				(used for temporary data involved in incomplete PCA calling)
	4. "project folder/data processing/data_init"				(the initialized data, ready to use, in .txt format)
	5. "project folder/genotype_450_dosage_matrix_qc"			(the organized dosage data)

