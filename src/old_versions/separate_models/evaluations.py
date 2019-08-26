import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import os

#===========================================================#

def nb_to_region (region):
	for i in range (len (region)):
		if region[i] == "1":
			region[i] = "Fusiform Gyrus"
		if region[i] == "2":
			region[i] = "left motor cortex"

		if region[i] == "3":
			region[i] = "right motor cortex"
		if region[i] == "4":
			region[i] = "left superior temporal sulcus"

		if region[i] == "5":
			region[i] = "right superior temporal sulcus"
		if region[i] == "6":
			region[i] = "Left Frontal Pole"

		if region[i] == "7":
			region[i] = "Right Frontal Pole"
		if region[i] == "8":
			region[i] = "Left Ventromedial PFC"
		if region[i] == "9":
			region[i] = "Right Ventromedial PFC"

#===========================================================#

def model_color (model_name):
	if "univ" in model_name:
		return "darkorange"
	elif "SVM" in model_name:
		return "blue"
	elif "RIDGE" in model_name:
		return "grey"
	elif "LASSO" in model_name:
		return "red"
	elif "RF" in model_name:
		return "darkgreen"

#===========================================================#

def get_eval (data, regions, label):

	data = data [data. Region.isin (regions)][label]

	return data.values

#===========================================================#

def get_models_names (evaluation_files):
	models_names = []
	for file in evaluation_files:
		if "univariate" in file:
			models_names. append ("univ_" + file. split('/')[-1]. split ('.')[0]. split ('_')[1])
		else:
			models_names. append (file. split('/')[-1]. split ('.')[0]. split ('_')[1])

	return models_names

#===========================================================#

def get_model_name (file):

	if "univariate" in file:
		model_name = "univ_" + file. split('/')[-1]. split ('.')[0]. split ('_')[1]
	else:
		model_name = file. split('/')[-1]. split ('.')[0]. split ('_')[1]

	return model_name

#===========================================================#

def process_one_subject (subject, measures):


	os. system ("rm separate_results/sub-%02d/*.pdf*"%subject)
	for conv in ["HH", "HR"]:

		evaluation_files = glob ("separate_results/sub-%02d/*%s.csv*"%(subject, conv))
		fig, ax = plt.subplots (nrows = len (measures), ncols = 1, figsize=(15,10))

		for file in evaluation_files:
			data = pd.read_csv (file, sep = ';', header = 0, na_filter = False, index_col=False)
			data = data.loc[data.groupby("Region")[measures[2]].idxmax(), :]
			data. sort_index(inplace=True)

			regions = data .loc[:,"Region"]. tolist ()
			x_names = [region.split ('_')[-1] for region in regions]
			#x_names = x_names. sort ()
			x_names. sort ()

			for i in range (len (measures)):
				evaluations = get_eval (data, regions, measures[i])
				model_name = get_model_name (file)

				if i == (len (measures) - 1):
					nb_to_region (x_names)

				ax[i]. plot (x_names, evaluations, label = model_name, marker = '.', color = model_color (model_name))

				ax[i]. set_ylabel (measures [i])
				ax[i]. set_xlabel ("Regions")
				ax[i]. legend ()
				ax[i].grid(which='major', linestyle=':', linewidth='0.25', color='black')
		plt. savefig ("separate_results/sub-%02d/eval_%s.pdf"%(subject, conv))
		plt. show ()

#===========================================================#

def process_multiple_subjects (subjects, measures, conv):

	evaluation_files = []
	for subject in subjects:
		try:
			files = sorted (glob ("separate_results/%s/*%s.csv*"%(subject, conv)))
			evaluation_files. append (files)
		except:
			print ("error in subject %s : %d"%(subject, ValueError))
			evaluation_files. append ([])
	return evaluation_files


#===========================================================#

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject")
	args = parser.parse_args()

	measures = ["recall", "precision", "f1_score"]

	# Group data by max of fscore to find the best set of predictive variables
	for sub in glob ("separate_results/*sub*"):
		csv_files = glob ("%s/*.csv"%sub)
		for csv_file in csv_files:
			data = pd. read_csv (csv_file, sep = ';', header = 0, na_filter = False, index_col=False)
			data = data.loc[data.groupby("Region")[measures[2]].idxmax(), :]
			data. sort_index(inplace=True)
			data.to_csv (csv_file, sep = ';', columns = data.columns, index = False)

	#exit (1)

	if args. subject in ["all", "ALL"]:

		sub_folders = glob ("separate_results/*sub*")
		subjects = []
		for sub in sub_folders:
			subjects. append (sub.split ('/')[-1])

		'''for subject in subjects:
			process_one_subject (int (subject. split ('-')[-1]), measures)'''

		for conv in ["HH", "HR"]:

			df_files = process_multiple_subjects (subjects, measures, conv)

			nb_models = len (df_files [1])
			fig, ax = plt.subplots (nrows = len (measures), ncols = 1, figsize = (20,10))
			dict = {}

			for j in range (nb_models):
				model_name = get_model_name (df_files [0][j])

				data =  pd.DataFrame ()
				for i in range (len (subjects)):
					if data. shape [0] == 0:
						data = pd. read_csv (df_files [i][j], sep = ';', header = 0, na_filter = False, index_col=False)

					else:
						df = pd. read_csv (df_files [i][j], sep = ';', header = 0, na_filter = False, index_col=False)
						for measure in measures:
							data [measure] = data [measure] + df [measure]

				data [measures] /= len (subjects)

				regions = data .loc[:,"Region"]. tolist (). sort ()
				x_names = [region.split ('_')[-1] for region in regions]
				#nb_to_region (x_names)


				for k in range (len (measures)):
					eval =  get_eval (data, regions, measures[k])
					if k == (len (measures) - 1):
						nb_to_region (x_names)
					ax[k]. plot (x_names, eval, label = model_name, marker = '.', color = model_color (model_name))
					ax[k]. set_ylabel (measures [k])
					#ax[k]. set_xlabel ("Regions")
					ax[k].grid(which='major', linestyle=':', linewidth='0.25', color='black')
					#ax[k]. set_size (15, 3)

				ax[2]. legend ()
				plt.xticks(rotation=60)


			#plt.text(0.2, 0.2, "73: cortex auditif primaire_gauche", fontsize=14)
			plt. savefig ("separate_results/eval_%s.pdf"%(conv), bbox_inches='tight')
			plt. show ()

	else:
		process_one_subject (int (args. subject), measures)















#=============================================================#
