import sys
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import numpy as np
import pandas as pd

import random as rd


from tools import toSuppervisedData, get_behavioral_data

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

#----------------------------------------------------
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


#------------------------------------
def get_max_of_list (data):
	best_model = data [0]
	for line in data[1:]:
		if np.mean (line[2]) > np. mean (best_model [2]):
			best_model = line
		elif (np.mean (line[2]) + 0.001 >= np. mean (best_model [2])) and line[1] < best_model[1]:
			best_model = line

	return best_model


#-------------------------------------
def cross_validation (convers,  target_index, target_column, subject, predictors, external_predictors, lag, lag_max, add_target = True, model = "SVM"):
	# First, rearange conversations randomly
	convers_order = rd. sample (range (len (convers)), k = len (convers))

	# testing the model on one conversation each time, and train it oh the others
	score_hh = []
	score_hr = []
	score = []

	for i in range (len (convers_order)):

		train_convers = [convers [j] for j in convers_order]
		valid_convers = convers [convers_order [i]]
		train_convers. remove (valid_convers)

		model = train_random_forrest (train_convers, target_column, target_index, subject, predictors,
									  external_predictors, lag, add_target = True)
		score. append (test_random_forest (model, valid_convers, target_index, target_column, subject,
									predictors, external_predictors, lag, lag_max = lag_max, add_target = True))

	results = np. mean (score, axis = 0)
	return (results)

#-----------------------------------------------------------------
def kfold_cross_validation (convers, target_column, target_index, subject, predictors, behavioral_predictors, lag_max, k = 1, add_target = True, model = "SVM"):

	# Split the data into k sets
	sets = []
	nb_convers_in_set = int (len (convers) / k)

	for i in range (0, len (convers), nb_convers_in_set):
		sets. append (convers [i : i + nb_convers_in_set])

	models_params = generate_tuples (behavioral_predictors, range (1, lag_max + 1))
	models_results = []

	# Set training data and test data
	# Test conversations, the last one from each set
	convers_test = []
	for i in range (k):
		if ((i % 2) == 0):
			convers_test. append (sets[i][-1])
		else:
			convers_test. append (sets[i][-2])

	# Training data
	convers_train = convers [:]
	for conv in convers_test:
		convers_train. remove (conv)


	for params in models_params:
		models_results. append (params + [[0, 0, 0]])

	# k-cross validation over all the sets
	for  i in range (len (models_params)):
		external_predictors, lag = models_params [i]

		for j in range (k):
			conv_train_set = sets[j][:]
			conv_train_set. remove (convers_test[j])


			results = cross_validation  (conv_train_set, target_index, target_column, subject, predictors,
										external_predictors, lag, lag_max = lag_max, add_target = True, model = model)
			for l in range (3):
			 	models_results[i][2][l] += results [l]

	best_model = get_max_of_list (models_results)

	# Evaluate the best model on test data
	# First we train the model on all training and validation data


	score_hh = []
	score_hr = []

	model = train_random_forrest (convers_train, target_column, target_index, subject, predictors,
								  best_model [0], best_model [1], add_target = True, model = model)

	for conv in convers_test:
		score = test_random_forest (model, conv, target_index, target_column, subject, predictors,
									best_model [0], best_model [1], lag_max = lag_max, add_target = True)

		if int (conv. split ('_')[-1]) % 2 == 1:
			score_hh. append (score)

		else:
			score_hr. append (score)

	return (best_model, [np. mean (score_hh, axis = 0), np. mean (score_hr, axis = 0)])


#----------------------------------------------------
#----- Fit and train random forrest
def  train_random_forrest (convers, target_column, target_index, subject, predictors, external_predictors, lag, add_target = True, model = "SVM"):

	X_all = np.empty ([0, 0])
	Y_all = np.empty ([0])
	# Online model updating on  the rest of conversations
	for conv in convers:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		if external_predictors != None:
			external_data = get_behavioral_data (subject, conv, external_predictors)
			data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
		else:
			data = pd.read_pickle (filename)[[target_column] + predictors]. values

		# normalize data
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)

		supervised_data = toSuppervisedData (data, lag, add_target = add_target)
		X = supervised_data.data
		Y = supervised_data.targets [:,target_index]

		if X_all. size == 0:
			X_all = X
			Y_all = Y
		else:
			X_all = np. concatenate ((X_all,X), axis = 0)
			Y_all = np. concatenate ((Y_all, Y), axis = 0)

	#X, Y = make_classification (n_samples = X. shape[0], n_features = X. shape[1], n_classes = 2)
	if model == "RF":
		clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

	elif model == "SVM":
		clf = svm. SVC (gamma='scale')

	clf. fit (X, Y)

	return clf

#-----------------------------------------------------------------------
def  test_random_forest (model, conv, target_index, target_column, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0, lag_max=5, add_target = True):
	results = []

	filename = "time_series/%s/discretized_physio_ts/%s.pkl" %(subject, conv)

	if not os.path.exists (filename):
		print ("file does not exist")

	if external_predictors != None:
		external_data = get_behavioral_data (subject, conv, external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]. values

	scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data = scaler. transform (data)

	supervised_data = toSuppervisedData (data, lag, add_target = add_target)
	X = supervised_data.data [lag_max - lag:, :]
	Y = supervised_data.targets [lag_max - lag:,target_index]

	pred = model. predict (X)
	real = Y

	score = [recall_score (real, pred), precision_score (real, pred), f1_score (real, pred)]

	return score
