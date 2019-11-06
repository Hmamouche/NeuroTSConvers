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
from sklearn.utils import shuffle

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split as train_test
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, TimeSeriesSplit

from src.feature_selection.reduction import manual_selection
from sklearn.cluster import KMeans, DBSCAN

from prediction import extract_models_params_from_crossv, features_in_select_results

#============================================================

def predict_area (subjects, target_column, set_of_behavioral_predictors, convers, lag, model, filename, find_params = False, method = "None"):

	"""
		- subjects:
		- target_column:
		- set_of_behavioral_predictors:
		- convers: the list of conversations
		- lag: the lag parameter
		- model: the prediction model name
		- filename: where to put the results
		- find_params: if TRUE, a k-fold-cross-validation  is used to find the parameters of the models, else using the previous one stored.
		- method: the feature selection method. None for no feature selection, and rfe for recursive feature elimination.
	"""

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

	if find_params:
		numb_test = 1
	else:
		numb_test = 1

	if model == "baseline":
		find_params = True

	for behavioral_predictors in set_of_behavioral_predictors:
		# concatenate data of all subjects  with regards to thje behavioral variables
		score = []
		for subject in subjects:

			all_data = concat_ (subject, target_column, convers, lag, behavioral_predictors, False, reg_model)

			if  np.isnan (all_data). any ():
				print ("Error in region %s with features %s"%(target_column,str (behavioral_predictors)))
				exit (1)

			''' names of lagged variables '''
			lagged_names = get_lagged_colnames (behavioral_predictors, lag)
			all_data = shuffle (all_data, random_state = 5)

			'''make 5 experiment of prediction the test data (20% of the data)'''
			perc = int (all_data. shape [0] * 0.2)

			for l in range (numb_test):
				test_index = list (range (l * perc,  (l + 1)* perc))
				train_index = list (range (l * perc)) + list (range ((l + 1) * perc,  all_data. shape [0]))

				train_data = all_data [train_index]
				test_data = all_data [test_index]

				# normalize data
				min_max_scaler = preprocessing. MinMaxScaler ()
				train_data [:,1:] = min_max_scaler. fit_transform (train_data [:,1:])
				test_data [:,1:] = min_max_scaler. transform (test_data [:,1:])

				''' check if feature selection must be used '''
				if method == "rfe" and model != "baseline":
					selected_indices = features_in_select_results (selection_results, target_column, lagged_names)
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

				# exception for lstm model: execute it without cross validation
				elif model == 'LSTM':
					best_model_params =  {'epochs': [20],  'neurons' : [30]}
					pred_model = train_model (train_data, model, best_model_params, lag)

				elif model == 'FUZZY':
					#pred_model = MultimodalEvolutionaryClassifier()
					pred_model = FuzzyPatternTreeTopDownClassifier ()
					pred_model. fit (train_data[:, 1:], train_data[:, 0])
					best_model_params = {}

				else:
					# extract model params from the previous k-fold-validation results
					models_params_file = glob. glob ("results/models_params/*%s_%s.csv" %(model, convers_type))[0]
					best_model_params = extract_models_params_from_crossv (models_params_file, target_column, lagged_names)
					# Train the model
					pred_model = train_model (train_data, model, best_model_params, lag)

				# Compute the score on test data
				score. append (test_model (test_data[: , 1:], test_data[: , 0], pred_model, lag, model))


		row = [target_column, method, best_model_params, str (dict(behavioral_predictors)), str (lagged_names),  str ([lagged_names [i] for i in selected_indices])] \
		+ np. mean (score, axis = 0). tolist () + np. std (score, axis = 0). tolist ()

		write_line (filename, row, mode = "a+")

		""" if the model the baseline (random), using multiple behavioral predictors has no effect """
		if model == "baseline":
			break

#====================================================================#

def predict_all (subjects, _regions, lag, k, model, remove, _find_params):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "models_params", "predictor_dict", "predictors_list", "selected_indices",
				"recall. mean", "precision. mean", "fscore. mean",  "recall. std", "precision. std", "fscore. std"]

	print (_regions)
	#exit (1)
	subjects_str = "subject"
	for subj in subjects:
		subjects_str += "_%s"%subj

	if _find_params:
		filename_hh = "results/models_params/%s_HH.csv"%(model)
		filename_hr = "results/models_params/%s_HR.csv"%(model)

	else:
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

	#_regions = ["region_%d"%i for i in regions]
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
	Parallel (n_jobs=2) (delayed (predict_area)
	(subjects, target_column, manual_selection (target_column), convers = hh_convers, lag = int (lag), model = model, filename = filename_hh, find_params = _find_params, method = "None")
									for target_column in _regions)

	Parallel (n_jobs=2) (delayed (predict_area)
	(subjects, target_column, manual_selection (target_column), convers = hr_convers, lag = int (lag), model = model, filename = filename_hr, find_params = _find_params, method = "None")
									for target_column in _regions)
