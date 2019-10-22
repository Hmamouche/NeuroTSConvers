# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche
import sys
import os
import glob

from sklearn import preprocessing
from sklearn.model_selection import train_test_split as train_test
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, TimeSeriesSplit
from sklearn.utils import shuffle

from ast import literal_eval
from joblib import Parallel, delayed

from src. feature_selection. reduction import manual_selection, reduce_train_test, ref_local
#from src. clustering import *
from src. prediction. tools import *
from src. prediction. training import *

from sklearn.ensemble import IsolationForest

from fylearn.garules import MultimodalEvolutionaryClassifier
from fylearn.nfpc import FuzzyPatternClassifier
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier

import warnings
warnings.filterwarnings("ignore")

#===========================================================
def extract_models_params_from_crossv (crossv_results_filename, brain_area, features):
	"""
		- extract  parameters of the mode from cross-validation results

		- crossv_results_filename: the filename of the model where the results are saved.
		- brain_area: brain region name
		- features: the set of predictive features
		- dictionary containing the parameter of the model
	"""

	features_exist_in_models_params = False
	models_params = pd.read_csv (crossv_results_filename, sep = ';', header = 0, na_filter = False, index_col = False)
	models_params = models_params. loc [(models_params ["region"] ==  brain_area)]

	# find the models_paras associated to each predictors_list
	for i in list (models_params. index):
		if set (literal_eval(models_params. loc [i, "predictors_list"])) == set (features):
			features_exist_in_models_params = True
			best_model_params = models_params. ix [i, "models_params"]
			break

	# else, choose the best model_params without considering features
	if not features_exist_in_models_params:
		best_model_params =  models_params. loc [models_params ["region"] ==  brain_area]
		best_model_params_index = best_model_params ["fscore. mean"].idxmax ()
		best_model_params = models_params. ix [best_model_params_index, "models_params"]#. iloc [0]

	return literal_eval (best_model_params)

#===========================================================
def features_in_select_results (selec_results, region, features):
	"""
		- check if a set of predictive variables has been processed in the feature selection step
			if so, the reduced form of this set is used
		- selec_results: the dataframe containing the feature selection results
			region: brain region
		- features: the set of predictive features
		- returns the indices (in the list features) of the selected variables
	"""
	features_exist = False
	results_region = selec_results. loc [(selec_results ["region"] ==  region)]
	selected_indices = [i for i in range (len (features))]

	# TODO: groupe by [region, features]

	''' find the models_paras associated to each predictors_list '''
	for i in list (results_region. index):
		if set (literal_eval(results_region. loc [i, "features"])) == set (features):
			features_exist = True
			selected_indices = literal_eval (results_region. ix [i, "selected_features"])
			break
	return selected_indices

#============================================================

def predict_area (subjects, target_column, set_of_behavioral_predictors, convers, lag, model, filename, find_params = False, method = "PCA"):

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
		numb_test = 5

	if model == "baseline":
		find_params = True
	for behavioral_predictors in set_of_behavioral_predictors:
		# concatenate data of all subjects  with regards to thje behavioral variables
		score = []

		all_data = concat_ (subjects [0], target_column, convers, lag, behavioral_predictors, False, reg_model)
		for subject in subjects [1:]:
			subject_data = concat_ (subject, target_column, convers, lag, behavioral_predictors, False, reg_model)
			all_data = np.concatenate ((all_data, subject_data), axis = 0)

		if  np.isnan (all_data). any ():
			print ("Error in region %s with features %s"%(target_column,str (behavioral_predictors)))
			exit (1)

		# names of lagged variables
		lagged_names = get_lagged_colnames (behavioral_predictors, lag)

		# outlier elimination
		outlier_model = IsolationForest (n_estimators=10, contamination = 0.2, behaviour = 'new')
		outlier_model. fit (all_data [:,1:])
		scores = outlier_model. predict (all_data [:,1:])

		delt = []
		for m in range (len (scores)):
		    if scores [m] == -1:
		        delt. append (m)

		all_data = np. delete (all_data, delt, axis = 0)

		# shuffle data
		#all_data = shuffle_data_by_blocks (all_data, 45)

		#np. random. shuffle (all_data)
		all_data = shuffle (all_data, random_state = 5)

		# make 5 experiment of prediction the test data (20% of the data)
		perc = int (all_data. shape [0] * 0.2)

		# tested number of variables for feature selection
		if model == "baseline":
			set_k = [all_data. shape [1] - 1]
		else:
			set_k = [3, 5, 6] #, 7, 9, 10 , 11, 12]

		for n_comp in set_k:
			print ("%s K = %s ----------"%(method, n_comp))
			score = []

			if type(n_comp) == int and n_comp >= all_data. shape [1]:
				break
			for l in range (numb_test):
				test_index = list (range (l * perc,  (l + 1)* perc))
				train_index = list (range (l * perc)) + list (range ((l + 1) * perc,  all_data. shape [0]))

				_train_data = all_data [train_index, :]
				_test_data = all_data [test_index, :]

				# normalization
				min_max_scaler = preprocessing. MinMaxScaler ()
				_train_data [:,1:] = min_max_scaler. fit_transform (_train_data [:,1:])
				_test_data [:,1:] = min_max_scaler. transform (_test_data [:,1:])


				if method == "RFE":
					train_data, test_data, selected_indices = ref_local (_train_data, _test_data, n_comp)
				else:
					train_data, test_data = reduce_train_test (_train_data, _test_data, method, perc_comps = float (n_comp) / (_train_data. shape[1] - 1), n_comps = n_comp)
					selected_indices = [i for i in range (len (lagged_names))]

				# k-fold cross validation
				if find_params and model not in ["LSTM", "FUZZY"]:
					valid_size = int (all_data. shape [0] * 0.2)
					# k_l_fold_cross_validation to find the best parameters
					best_model_params, pred_model = k_l_fold_cross_validation (train_data, target_column, model, lag = 1, n_splits = 1, block_size = valid_size)

				# exception for the lstm model: execute it without cross validation because of time ...
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

			row = [target_column, "%s_%s"%(method, str (n_comp)), best_model_params, str (dict(behavioral_predictors)), str (lagged_names),  str ([lagged_names [i] for i in selected_indices])] \
			+ np. mean (score, axis = 0). tolist () + np. std (score, axis = 0). tolist ()

			write_line (filename, row, mode = "a+")

			# if the model the baseline (random), using multiple behavioral predictors has no effect
			if model == "baseline":
				break

		if model == "baseline":
			break

#====================================================================#

def predict_all (subjects, _regions, lag, k, model, remove, _find_params):

	print ("-- MODEL :", model)
	colnames = ["region", "dm_method", "models_params", "predictors_dict", "predictors_list", "selected_predictors",
				"recall. mean", "precision. mean", "fscore. mean",  "recall. std", "precision. std", "fscore. std"]

	print (_regions)
	subjects = ["sub-%02d"%i for i in subjects]

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
	Parallel (n_jobs=5) (delayed (predict_area)
	(subjects, target_column, manual_selection (target_column), convers = hh_convers, lag = int (lag), model = model, filename = filename_hh, find_params = _find_params)
									for target_column in _regions)

	Parallel (n_jobs=5) (delayed (predict_area)
	(subjects, target_column, manual_selection (target_column), convers = hr_convers, lag = int (lag), model = model, filename = filename_hr, find_params = _find_params)
									for target_column in _regions)
