# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche
import sys
import os
import glob

from src. clustering import *
from src. prediction. tools import *
from src. prediction. training import *
from sklearn import preprocessing
from ast import literal_eval

from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split as train_test
from sklearn.model_selection import StratifiedShuffleSplit

from src.reduction import manual_selection
from sklearn.cluster import KMeans, DBSCAN

#============================================================

def predict_area (subjects, target_column, set_of_behavioral_predictors, convers, lag, model, filename, find_params = False, method = "None"):

	if (int (convers[0]. split ('_')[-1]) % 2 == 1): convers_type = "HH"
	else : convers_type = "HR"

	# Extract the selected features from features sselection results
	if os.path.exists ("results/selection/selection_%s.csv" %(convers_type)):
		selection_results = pd.read_csv ("results/selection/selection_%s.csv" %(convers_type),  sep = ';', header = 0, na_filter = False, index_col=False)

	for behavioral_predictors in set_of_behavioral_predictors:

		# concatenate data of all subjects  with regards to thje behavioral variables
		score = []
		all_data = []
		all_data = concat_ (subjects[0], target_column, convers, lag, behavioral_predictors, False)


		for subject in subjects[1:]:
			subject_data = concat_ (subject, target_column, convers, lag, behavioral_predictors, False)
			all_data = np.concatenate ((all_data, subject_data), axis = 0)

		if  np.isnan (all_data). any ():
			continue

		#print (pd. DataFrame (all_data))
		#exit (1)

		# lagged variables names
		lagged_names = get_lagged_colnames (behavioral_predictors)

		""" split the data into test and training sets with stratified way """
		sss = StratifiedShuffleSplit (n_splits = 1, test_size = 0.2, random_state = 5)
		for train_index, test_index in sss.split (all_data[:, 1:], all_data [:, 0]):
			train_data = all_data [train_index]
			test_data = all_data [test_index]
			break

		# normalize data
		min_max_scaler = preprocessing. MinMaxScaler ()
		train_data [:,1:] = min_max_scaler. fit_transform (train_data [:,1:])
		test_data [:,1:] = min_max_scaler. transform (test_data [:,1:])

		'''clust_, nk = kmeans_auto (train_data [:, 1:2], max_k = 70)
		print (nk)
		exit (1)'''

		# clustering the data
		'''for j in range (1, all_data. shape [1]):

			clustering = KMeans (n_clusters = 8). fit (train_data [:, j : j+1])
			#clustering = DBSCAN (eps=3, min_samples=2).fit (train_data [:, j : j+1])
			train_data [:, j] = clustering. labels_
			test_data [:, j] = clustering. fit_predict (test_data [:, j : j+1])'''

		'''print (all_data. shape)
		print (train_data. shape [0]/ all_data. shape [0])
		print (pd. DataFrame (train_data))
		exit (1)'''


		if method == "rfe" and model != "random":
			# Feature selection
			selected_indices = selection_results . loc [selection_results ["features"] == str (lagged_names)] ["selected_features"]. values
			selected_features = selection_results . loc [selection_results ["features"] == str (lagged_names)] ["features"]. values

			if len (selected_indices) > 0:
				# from str to python object (list in this case)
				selected_indices = literal_eval (selected_indices[0])
				selected_features = selected_features [0]
				train_data = train_data [:, [0] + [int (a + 1) for a in selected_indices]]
				test_data = test_data [:, [0] + [int (a + 1) for a in selected_indices]]
			else:
				selected_features = str (lagged_names)
				selected_indices = str ([x for x in range (len (lagged_names))])

		elif method == "None" or model == "random":
			selected_features = str (lagged_names)
			selected_indices = str ([x for x in range (len (lagged_names))])
			method == "None"

		# k-fold cross validation
		if find_params:
			valid_size = int (all_data. shape [0] * 0.2)
			# k_l_fold_cross_validation to find the best parameters
			best_model_params, pred_model = k_l_fold_cross_validation (train_data, target_column, model, lag = 1, n_splits = 1, block_size = valid_size)


		else:
			model_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type))[0]
			models_params = pd.read_csv (model_file, sep = ';', header = 0, na_filter = False, index_col = False)
			#best_model_params = models_params. loc [models_params ["region"] ==  target_column] ["models_params"]. iloc [0]
			#best_model_params = models_params.loc [models_params. groupby ("region") ["fscore"].idxmax (), :]
			best_model_params = models_params. loc [(models_params ["region"] ==  target_column) & (models_params["predictors"] == str (behavioral_predictors))] ["models_params"] #. iloc [0]

			if best_model_params. shape [0] == 0:
				best_model_params = models_params. loc [models_params ["region"] ==  target_column] ["models_params"]. iloc [0]
				best_model_params = models_params.loc [models_params. groupby ("region") ["fscore"].idxmax (), "models_params"]. iloc [0]
			else:
				best_model_params = best_model_params. iloc [0]

			best_model_params = literal_eval (best_model_params)

			# Train the model
			pred_model = train_model (train_data, model, best_model_params, lag)

		# Compute the score on test data
		score = test_model (test_data[: , 1:], test_data[: , 0], pred_model, lag, model)

		row = [target_column, method, best_model_params, str (dict (behavioral_predictors)), selected_features, selected_indices] + score
		write_line (filename, row, mode = "a+")

		""" if the model the baseline (random), using multiple behavioral predictors has no effect """
		if model == "random":
			break

#====================================================================#

def predict_all (subjects, regions, lag, k, model, remove):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "models_params", "predictors", "selected_predictors", "selected_indices", "recall", "precision", "fscore"]

	subjects_str = "subject"
	for subj in subjects:
		subjects_str += "_%s"%subj

	filename_hh = "results/prediction/%s_HH.csv"%(model)
	filename_hr = "results/prediction/%s_HR.csv"%(model)

	for filename in [filename_hh, filename_hr]:
		# remove previous output files ir remove == true
		if remove:
			os. system ("rm %s"%filename)
		if not os.path.exists (filename):
			f = open (filename, "w+")
			f.write (';'. join (colnames))
			f. write ('\n')
			f. close ()

	_regions = ["region_%d"%i for i in regions]
	subjects = ["sub-%02d"%i for i in subjects]

	# fill HH and HR conversations
	convers = list_convers ()
	hh_convers = []
	hr_convers = []

	for i in range (len (convers)):
		if i % 2 == 1:
			hr_convers. append (convers [i])
		else:
			hh_convers. append (convers [i])

	# Predict HH  and HR conversations separetely
	Parallel (n_jobs=5) (delayed (predict_area) (subjects, target_column, manual_selection (target_column), convers = hh_convers, lag = int (lag), model = model, filename = filename_hh)
									for target_column in _regions)

	Parallel (n_jobs=5) (delayed (predict_area) (subjects, target_column, manual_selection (target_column), convers = hr_convers, lag = int (lag), model = model, filename = filename_hr)
									for target_column in _regions)
