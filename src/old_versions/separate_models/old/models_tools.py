import sys
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import numpy as np
import pandas as pd

import random as rd
from sklearn import linear_model
from scipy.signal import find_peaks


from tools import toSuppervisedData, get_behavioral_data

import itertools as it

#===========================================
def find_peaks_ (y, height = 0):
	x = []
	for i in range (len (y)):
		if y[i] > height:
			x.append (i)
	return x

#===========================================
def discretize (x):
	result = [0 for i in range (len (x))]
	peaks = find_peaks_ (x, height=0)

	for i in peaks:
		result [i] = 1

	return result
#--------------------------------#
#----+     NORMALIZE      +------#
#--------------------------------#
def normalize (M):
    minMax = np.empty ([M.shape[1], 2])
    for i in range(M.shape[1]):
        #print (M[:,i])
        max = np.max(M[:,i])
        min = np.min(M[:,i])
        minMax[i,0] = min
        minMax[i,1] = max

        for j in range(M.shape[0]):
            M[j,i] = (M[j,i] - min) / (max - min)

    return minMax

#--------------------------------------#
#----+     DENORMALIZATION      +------#
#--------------------------------------#
def inverse_normalize (M,minMax):
    for i in range(M.shape[0]):
        M[i] = M[i] * (minMax[1] - minMax[0]) + minMax[0]

#--------------------------------------------------
def printf (l):
	for a in l:
		print (a)
	print ('----------')
#----------------------------------------------------
# generate tuples from two lists
def generate_tuples (a, b):
	c = [[x , y] for x in a for y in b]
	return c

#===================================================
def generate_models (params):

	combinations = it. product (* (params[Name] for Name in params. keys ()))

	keys = list (params. keys ())
	res = []
	for combin in list (combinations):
		dict = {}
		for i in range (len (keys)):
			dict[keys[i]] = combin [i]
		res. append (dict)

	return res

#===================================================
#---- get items from dict as string
def get_items (predictors, external_predictors):
	if predictors != None:
		external_variables = []
		for key in external_predictors. keys ():
			external_variables += external_predictors[key]
		variables = '+'.join (predictors + external_variables)
	else:
		variables = '+'.join (predictors)
	return (variables)


#===================================================
def get_max_of_list (data):
	best_model = data [0]
	best_index = 0
	i = 1
	for line in data[1:]:
		if np.mean (line[1]) > np. mean (best_model [1]):
			best_model = line
			best_index = i
		i += 1

	return best_model, best_index

#-------------------------------------
def k_fold_cross_validation (convers, target_column, subject, predictors, external_predictors, lag, lag_max, add_target, model, params):

	# First, rearange conversations randomly
	convers_order = rd. sample (range (len (convers)), k = len (convers))

	# testing the model on one conversation each time, and train it oh the others
	score_hh = []
	score_hr = []
	score = []
	target_index = 0

	for i in range (len (convers_order)):
		train_convers = [convers [j] for j in convers_order]
		valid_convers = convers [convers_order [i]]
		train_convers. remove (valid_convers)

		pred_model = train_model (model, params, train_convers, target_column, subject, predictors,
		external_predictors, lag, add_target = add_target)

		score. append (test_model (pred_model, valid_convers, target_column, subject,
									predictors, external_predictors, lag, lag_max = lag_max, add_target = add_target))

	results = np. mean (score, axis = 0)
	return (results)

#-----------------------------------------------------------------
def k_l_fold_cross_validation (convers, target_column, subject, predictors, behavioral_predictors, lag_max, k, add_target, model):

	# Split the data into k sets
	sets = []
	nb_convers_in_set = int (len (convers) / k)
	target_index = 0

	for i in range (0, len (convers), nb_convers_in_set):
		sets. append (convers [i : i + nb_convers_in_set])

	# models_params = generate_tuples (behavioral_predictors, [lag_max])

	if model == "SVM":
		models_params = generate_models ({'C': [10, 100, 200, 500, 1000], 'kernel': ['rbf']})

	elif model in ["RIDGE", "Ridge", "LASSO", "Lasso"]:
		models_params = generate_models ({'alpha': [0.1, 0.01, 0.2, 0.3]})

	elif model == "RF":
		models_params = generate_models ({'bootstrap': [True, False],
		 'max_depth': [10, 50, 100],
		 'max_features': ['auto'],
		 'n_estimators': [10, 20, 50, 100, 150]})

	models_results = []

	# Set training data and test data
	# Test conversations, the last one from each set
	convers_test = []
	for i in range (k):
		convers_test. append (sets[i][-1])

	# Training data
	convers_train = convers [:]
	for conv in convers_test:
		convers_train. remove (conv)

	#print (convers_train, "\n------\n", convers_test)
	#exit (1)

	for params in models_params:
		models_results. append ([str (params)] + [[0, 0, 0]])

	if behavioral_predictors != None:
		external_predictors = behavioral_predictors. copy ()
	else:
		external_predictors = None

	lag = lag_max

	# k-fold-cross validation over all the blocks
	for i in range (len (models_params)):
		for j in range (k):
			conv_train_set = sets[j][:]
			conv_train_set. remove (convers_test[j])

			results = k_fold_cross_validation  (conv_train_set, target_column, subject, predictors,
										external_predictors, lag, lag_max = lag_max, add_target = add_target, model = model, params = models_params [i])
			for l in range (3):
			 	models_results[i][1][l] += results [l]

	# Get the best model parameters
	best_model, best_index = get_max_of_list (models_results)

	# Evaluate the best model on test data
	scores = []
	pred_model = train_model (model, models_params[best_index], convers_train, target_column, subject, predictors,
								  external_predictors, lag = lag, add_target = add_target)

	for conv in convers_test:
		score = test_model (pred_model, conv, target_column, subject, predictors,
									external_predictors, lag = lag, lag_max = lag_max, add_target = add_target)

		scores. append (score)

	return best_model[0], np. mean (scores, axis = 0). tolist ()

#==============================================================
#----- Fit and train random forrest
def  train_model (model, params, convers, target_column, subject, predictors, external_predictors, lag, add_target):

	X_all = np.empty ([0, 0])
	Y_all = np.empty ([0])
	target_index = 0

	# Online model updating on  the rest of conversations
	for conv in convers:
		if model in ["RIDGE", "Ridge", "LASSO", "Lasso"]:
			filename = "time_series/%s/new_physio_ts/%s.pkl"%(subject, conv)
		else:
			filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		if external_predictors != None:
			behav_data = get_behavioral_data (subject, conv, external_predictors)

			data = pd. concat ([pd.read_pickle (filename)[[target_column] + predictors], behav_data], axis = 1). values
		else:
			data = pd.read_pickle (filename)[[target_column] + predictors]. values


		supervised_data = toSuppervisedData (data, lag, add_target = add_target, nb_inter = len (predictors))
		X = supervised_data.data
		Y = supervised_data.targets [:,0]

		if X_all. size == 0:
			X_all = X
			Y_all = Y
		else:
			X_all = np. concatenate ((X_all,X), axis = 0)
			Y_all = np. concatenate ((Y_all, Y), axis = 0)

	# normalize data
	'''scaler = MinMaxScaler()
	scaler.fit_transform(X_all)
	X_all =  scaler. transform (X_all)'''

	if model == "RF":
		pred_model = RandomForestClassifier(**params)

	elif model == "SVM":
		pred_model = svm. SVC (**params)

	elif model in ["RIDGE", "Ridge"]:
		pred_model = linear_model.Ridge (**params)

	elif model in ["LASSO", "Lasso"]:
		pred_model = linear_model.Lasso (**params)

	pred_model. fit (X_all, Y_all)

	return pred_model

#-----------------------------------------------------------------------
def  test_model (model, conv, target_column, subject, predictors, external_predictors, lag, lag_max, add_target):
	results = []
	target_index = 0

	if model in ["RIDGE", "Ridge", "LASSO", "Lasso"]:
		filename = "time_series/%s/new_physio_ts/%s.pkl"%(subject, conv)
	else:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

	if not os.path.exists (filename):
		print ("file does not exist")


	if external_predictors != None:
		external_data = get_behavioral_data (subject, conv, external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values

	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]. values


	'''scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data = scaler. transform (data)'''

	supervised_data = toSuppervisedData (data, lag, add_target = add_target, nb_inter = len (predictors))
	X = supervised_data.data [lag_max - lag:, :]
	Y = supervised_data.targets [lag_max - lag:,0]

	# normalize data
	'''scaler = MinMaxScaler()
	scaler.fit_transform(X)
	X =  scaler. transform (X)'''

	pred = model. predict (X)
	real = Y

	if type(model).__name__ in ["Ridge", "RIDGE", "Lasso", "LinearRegression"]:
		pred = discretize (pred)
		real = discretize (real)

	score = [recall_score (real, pred, average = 'weighted'), precision_score (real, pred, average = 'weighted'), f1_score (real, pred, average = 'weighted')]

	return score
