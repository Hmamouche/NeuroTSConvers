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

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split as train_test
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, TimeSeriesSplit

from src.reduction import manual_selection
from sklearn.cluster import KMeans, DBSCAN


#===========================================================
def features_in_select_results (selec_results, region, features):
	features_exist = False
	results_region = selec_results. loc [(selec_results ["region"] ==  region)]
	selected_indices = [i for i in range (len (features))]

	''' find the models_paras associated to each predictors_list '''
	for i in list (results_region. index):
		if set (literal_eval(results_region. loc [i, "features"])) == set (features):
			features_exist = True
			selected_indices = literal_eval (results_region. ix [i, "selected_features"])
			break

	return selected_indices



#============================================================

def predict_area (subjects, target_column, set_of_behavioral_predictors, convers, lag, model, filename, find_params = False, method = "None"):

	#print (target_column)
	if (int (convers[0]. split ('_')[-1]) % 2 == 1): convers_type = "HH"
	else : convers_type = "HR"

	# Extract the selected features from features sselection results
	if os.path.exists ("results/selection/selection_%s.csv" %(convers_type)):
		selection_results = pd.read_csv ("results/selection/selection_%s.csv" %(convers_type),  sep = ';', header = 0, na_filter = False, index_col=False)


	if model in ["RIDGE", "LASSO"]:
		reg_model = True
	else:
		reg_model = False


	for behavioral_predictors in set_of_behavioral_predictors:
		# concatenate data of all subjects  with regards to thje behavioral variables
		score = []
		all_data = []
		all_data = concat_ (subjects [0], target_column, convers, lag, behavioral_predictors, False, reg_model)

		for subject in subjects [1:]:
			subject_data = concat_ (subject, target_column, convers, lag, behavioral_predictors, False, reg_model)
			all_data = np.concatenate ((all_data, subject_data), axis = 0)

		if  np.isnan (all_data). any ():
			continue

		#print (pd. DataFrame (all_data))
		#exit (1)
		# lagged variables names
		lagged_names = get_lagged_colnames (behavioral_predictors)


		""" split the data into test and training sets with stratified way """
		sss = ShuffleSplit (n_splits = 1, test_size = 0.2, random_state = 5)
		for train_index, test_index in sss.split (all_data[:, 1:], all_data [:, 0:1]):
			train_data = all_data [train_index]
			test_data = all_data [test_index]
			break

		# normalize data
		min_max_scaler = preprocessing. MinMaxScaler ()
		train_data [:,1:] = min_max_scaler. fit_transform (train_data [:,1:])
		test_data [:,1:] = min_max_scaler. transform (test_data [:,1:])

		#print (pd. DataFrame (test_data))
		#continue

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


		''' check if feature selection must be used '''
		if method == "rfe" and model != "baseline":
			#selected_indices = selection_results . loc [selection_results ["features"] == str (lagged_names)] ["selected_indices"]. values
			selected_indices = features_in_select_results (selection_results, target_column, lagged_names)
			#print (type (selected_indices))
			#print (selected_indices)
			#exit (1)
			train_data = train_data [:, [0] + [int (a + 1) for a in selected_indices]]
			test_data = test_data [:, [0] + [int (a + 1) for a in selected_indices]]



		elif method == "None" or model == "baseline":
			selected_features = str (lagged_names)
			selected_indices =  [x for x in range (len (lagged_names))]
			method == "None"

		# k-fold cross validation
		if find_params and model not in ["LSTM"]:
			valid_size = int (all_data. shape [0] * 0.2)
			# k_l_fold_cross_validation to find the best parameters
			best_model_params, pred_model = k_l_fold_cross_validation (train_data, target_column, model, lag = 1, n_splits = 1, block_size = valid_size)

		elif model == 'LSTM':
			best_model_params =  {'epochs': [20],  'neurons' : [30]}
			pred_model = train_model (train_data, model, best_model_params, lag)

		else:
			model_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type))[0]
			models_params = pd.read_csv (model_file, sep = ';', header = 0, na_filter = False, index_col = False)

			features_exist_in_models_params = False
			models_params = models_params. loc [(models_params ["region"] ==  target_column)]

			''' find the models_paras associated to each predictors_list '''
			for i in list (models_params. index):
				if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (lagged_names):
					features_exist_in_models_params = True
					best_model_params = models_params. ix [i, "models_params"]
					break

			''' else, choose the best model_params '''
			if   not features_exist_in_models_params:
				best_model_params =  models_params. loc [models_params ["region"] ==  target_column]
				best_model_params_index = best_model_params ["fscore"].idxmax ()
				best_model_params = models_params. ix [best_model_params_index, "models_params"]#. iloc [0]


			best_model_params = literal_eval (best_model_params)

			# Train the model
			pred_model = train_model (train_data, model, best_model_params, lag)

		# Compute the score on test data
		score = test_model (test_data[: , 1:], test_data[: , 0], pred_model, lag, model)

		row = [target_column, method, best_model_params, str (dict(behavioral_predictors)), str (lagged_names),  str ([lagged_names [i] for i in selected_indices])] + score
		write_line (filename, row, mode = "a+")

		""" if the model the baseline (random), using multiple behavioral predictors has no effect """
		if model == "baseline":
			break

#====================================================================#

def predict_all (subjects, regions, lag, k, model, remove):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "models_params", "predictor_dict", "predictors_list", "selected_indices", "recall", "precision", "fscore"]

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
	Parallel (n_jobs=1) (delayed (predict_area) (subjects, target_column, manual_selection (target_column), convers = hh_convers, lag = int (lag), model = model, filename = filename_hh)
									for target_column in _regions)

	Parallel (n_jobs=1) (delayed (predict_area) (subjects, target_column, manual_selection (target_column), convers = hr_convers, lag = int (lag), model = model, filename = filename_hr)
									for target_column in _regions)
