import pandas as pd
import matplotlib.pyplot as plt
import argparse


def get_rmse (data, left_regions, right_regions):

	data_left = data[data.Region.isin (left)]
	data_right = data[data.Region.isin (right)]
	
	data_left = data_left.loc[data_left.groupby("Region")["Rmse"].idxmin(), "Rmse"]
	data_right = data_right.loc[data_right.groupby("Region")["Rmse"].idxmin(), "Rmse"]
	
	data_left. sort_index(inplace=True)
	data_right. sort_index(inplace=True)
	
	return data_left.values, data_right.values
	
if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	args = parser.parse_args()
	
	var = pd.read_csv ("results/sub-%02d/predictions_HH.txt"%args.subject, sep = ';', header = 0, na_filter = False, index_col=False)
	arima = pd.read_csv ("results/sub-%02d/predictions_arima_HH.txt"%args.subject, sep = ';', header = 0, na_filter = False, index_col=False)
	ar = pd.read_csv ("results/sub-%02d/predictions_ar_HH.txt"%args.subject, sep = ';', header = 0, na_filter = False, index_col=False)

	
	left = ["region_73", "region_75", "region_79","region_87", "region_121", "region_123"]
	right = ["region_74", "region_76", "region_80","region_88", "region_122", "region_124"]

	arima_l, arima_r = get_rmse (arima, left, right)
	ar_l, ar_r = get_rmse (ar, left, right)
	var_l, var_r = get_rmse (var, left, right)
	
	
	# Plot the results
	fig, ax = plt.subplots (nrows = 2, ncols = 1, figsize=(10,6))

	ax[0]. plot (left, ar_l, label = "AR")
	ax[0]. plot (left, arima_l, label = "ARIMA")
	ax[0]. plot (left, var_l, label = "VAR")
	ax[0]. set_xlabel ("Left regions")
	ax[0]. set_ylabel ("RMSE")
	ax[0]. legend ()
	
	ax[1]. plot (right, ar_r, label = "AR")
	ax[1]. plot (right, arima_r, label = "ARIMA")
	ax[1]. plot (right, var_r, label = "VAR")
	ax[1]. set_xlabel ("Right regions")
	ax[0]. set_ylabel ("RMSE")
	ax[1]. legend ()
	#plt. show ()
	plt. savefig ("results/sub-%02d/eval.pdf"%args.subject)
