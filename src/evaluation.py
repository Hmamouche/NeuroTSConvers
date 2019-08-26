import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import os
import matplotlib.ticker as ticker

#===========================================================#

def nb_to_region (region):
	for i in range (len (region)):
		if region[i] == "1":
			region[i] = "FFA"
		if region[i] == "2":
			region[i] = "L-motor-cortex"

		if region[i] == "3":
			region[i] = "R-motor-cortex"
		if region[i] == "4":
			region[i] = "L-sup-tmp-sulcus"

		if region[i] == "5":
			region[i] = "R-sup-tmp-sulcus"
		if region[i] == "6":
			region[i] = "L-Frontal-Pole"

		if region[i] == "7":
			region[i] = "R-Frontal-Pole"
		if region[i] == "8":
			region[i] = "L-Ventromedial-PFC"
		if region[i] == "9":
			region[i] = "R-Ventromedial-PFC"

#===========================================================#

def model_color (model_name):
	if "GB" in model_name:
		return "indianred"
	elif "SVM" in model_name:
		return "darkorange"
	elif "RIDGE" in model_name:
		return "green"
	elif "LASSO" in model_name:
		return "red"
	elif "RF" in model_name:
		return "royalblue"
	elif "SGD" in model_name:
		return "lightcoral"
	elif "LSTM" in model_name:
		return "black"
	elif model_name == "random":
		return "grey"

#===========================================================#

def get_eval (data, regions, label):

	data = data [data. region.isin (regions)][label]
	return data.values

#===========================================================#

def get_models_names (evaluation_files):
	models_names = []

	for file in evaluation_files:
		models_names. append (file. split('/')[-1]. split ('_')[0])

	return models_names

#===========================================================#

def get_model_name (file):

	model_name = file. split('/')[-1]. split ('_')[0]
	return model_name

#===========================================================#

def process_multiple_subject (measures):

	os. system ("rm results/prediction/*.pdf*")
	for conv in ["HH", "HR"]:

		evaluation_files = glob ("results/prediction/*%s.csv*"%conv)
		evaluation_files. sort ()
		fig, ax = plt.subplots (nrows = len (measures), ncols = 1, figsize=(15,10))

		bar_with = 0.03
		distance = 0
		local_dist = 0.005

		for file in evaluation_files:
			data = pd.read_csv (file, sep = ';', header = 0, na_filter = False, index_col=False)

			if data. shape [0] == 0:
				continue

			'''data = data.loc[data.groupby("Regions")[measures[2]].idxmax(), :]
			data. sort_index(inplace=True)'''

			regions = data .loc[:,"region"]. tolist ()
			x_names = [int (region.split ('_')[-1]) +  distance for region in regions]
			regions_names = [region.split ('_')[-1] for region in regions]
			nb_to_region (regions_names)
			distance += bar_with + local_dist

			for i in range (len (measures)):
				evaluations = get_eval (data, regions, measures[i])
				model_name = get_model_name (file)

				#ax[i]. bar (x_names, evaluations, label = model_name, marker = '.', color = model_color (model_name))
				ax[i]. bar (x_names, evaluations, label = model_name, width = bar_with, capsize=7, color = model_color (model_name))
				ax[i]. set_ylabel (measures [i])
				ax[i]. set_xlabel ("Regions")

			for i in range (len (measures)):
				ax[i].xaxis. set_major_locator ((ticker. IndexLocator (base = 1, offset= 2 * bar_with)))
				ax[i].yaxis. set_major_locator (ticker. MultipleLocator (0.1))
				ax[i]. set_xticklabels (regions_names, minor = False)
				ax[i]. grid (which='major', linestyle=':', linewidth='0.25', color='black')

		plt.legend (loc='upper right', bbox_to_anchor = (1.1, 3.5), fancybox=True, shadow=True, ncol=1)
		plt. savefig ("results/prediction/eval_%s.pdf"%conv)
		plt. show ()

#===========================================================#

if __name__=='__main__':

	measures = ["recall", "precision", "fscore"]

	# Group data by max of fscore to find the best set of predictive variables
	csv_files = glob ("results/prediction/*.csv*")

	for csv_file in csv_files:
		data = pd. read_csv (csv_file, sep = ';', header = 0, na_filter = False, index_col=False)
		data = data.loc [data. groupby ("region") ["fscore"].idxmax (), :]
		data. sort_index (inplace=True)
		data.to_csv (csv_file, sep = ';', columns = data.columns, index = False)

	process_multiple_subject (measures)

#=============================================================#
