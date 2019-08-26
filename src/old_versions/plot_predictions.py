import pandas as pd
import matplotlib.pyplot as plt
import argparse

#--------------------------------------------------------------------
def get_eval (data, regions, label):

	data = data [data. Region.isin (regions)][label]
	#print (data)
	#exit (1)

	'''data = data.loc[data.groupby("Region")[label].idxmax(), label]

	data. sort_index(inplace=True)'''

	return data.values
#--------------------------------------------------------------------
'''def get_eval (data, left_regions, right_regions, label = "precision"):

	data_left = data[data.Region.isin (left)]
	data_right = data[data.Region.isin (right)]

	data_left = data_left.loc[data_left.groupby("Region")[label].idxmin(), label]
	data_right = data_right.loc[data_right.groupby("Region")[label].idxmin(), label]

	data_left. sort_index(inplace=True)
	data_right. sort_index(inplace=True)

	return data_left.values, data_right.values'''

#--------------------------------------------------------------------
if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	args = parser.parse_args()

	for conv in ["HH", "HR"]:
		univariate = pd.read_csv ("results/sub-%02d/univariate_RF_%s.csv" %(args.subject, conv), sep = ';', header = 0, na_filter = False, index_col=False)
		SVM = pd.read_csv ("results/sub-%02d/multivariate_SVM_%s.csv" %(args.subject, conv), sep = ';', header = 0, na_filter = False, index_col=False)
		RF = pd.read_csv ("results/sub-%02d/multivariate_RF_%s.csv" %(args.subject, conv), sep = ';', header = 0, na_filter = False, index_col=False)

		regions = ["region_73", "region_74", "region_75", "region_76", "region_79", "region_80", "region_87", "region_88", "region_121", "region_122", "region_123", "region_124"]


		evaluations = [[get_eval (univariate, regions, "recall"), get_eval (SVM, regions, "recall"), get_eval (RF, regions, "recall")],
						[get_eval (univariate, regions, "precision"), get_eval (SVM, regions, "precision"), get_eval (RF, regions, "precision")],
						[get_eval (univariate, regions, "f1_score"), get_eval (SVM, regions, "f1_score"), get_eval (RF, regions, "f1_score")]]

		ordre = ["univariate", "SVM", "RF"]

		# Plot the results
		fig, ax = plt.subplots (nrows = 3, ncols = 1, figsize=(15,10))

		measures = ["recall", "precision", "F-score"]

		for i in range (3):
			x_names = [region.split ('_')[-1] for region in regions]

			for j in range (3):
				ax[i]. plot (x_names, evaluations[i][j], label = ordre[j])

			ax[i]. set_ylabel (measures [i])
			ax[i]. legend ()

		plt. savefig ("results/sub-%02d/eval_%s.pdf"%(args.subject, conv))
		plt. show ()
