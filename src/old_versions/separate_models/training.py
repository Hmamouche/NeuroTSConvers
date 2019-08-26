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
	peaks = find_peaks_ (x, height=0.2)

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
def k_fold_cross_validation (data, model, params, nb_sets):

	score = []

	# split trai data into nb_sets blocks
	splits = np.split (data, nb_sets)

	# First, rearange conversations randomly
	convers_order = rd. sample (range (len (splits)), k = len (splits))

	# train and validate each time
	# validate the model on one conversation each time, and train it oh the others
	for i in range (len (convers_order)):
		train_convers = []

		# validation data
		validation_convers  = splits [convers_order [i]]

		# train data
		for j in range (len (convers_order)):
			if j != i:
				if len (train_convers) == 0:
					train_convers = splits [ convers_order [j] ]
				else:
					train_convers = np. concatenate ( (train_convers, splits [ convers_order [j] ] ),  axis = 0)

		pred_model = train_model (train_convers, model, params)

		score. append (test_model (validation_convers[:, 1:], validation_convers[:, 0], pred_model))

	results = np. mean (score, axis = 0)
	return (results)

#===================================================================

def k_l_fold_cross_validation (data, target_column, model, n, k, l):

	# models_params = generate_tuples (behavioral_predictors, [lag_max])
	if model == "SVM":
		models_params = generate_models ({'C': [10, 50, 100, 150], 'kernel': ['rbf']})

	elif model in ["RIDGE", "Ridge", "LASSO", "Lasso"]:
		models_params = generate_models ({'alpha': [0.1, 0.01, 0.2, 0.3]})

	elif model == "RF":
		models_params = generate_models ({'bootstrap': [True],
		 'max_depth': [10],
		 'max_features': ['auto'],
		 'n_estimators': [50, 100, 150]})

	models_results = []

	# Split the data into k sets
	splits = np.split (data, l)

	nb_of_test_data_by_split = k * n

	test_data = []
	train_data = []

	for params in models_params:
		models_results. append ([str (params)] + [[0, 0, 0]])

	# Deal with each split
	for split in splits:
		# define training and test data
		if len (test_data) == 0:
			test_data = split[-nb_of_test_data_by_split : ]

		else:
			test_data = np. concatenate ((test_data, split[-nb_of_test_data_by_split : ]))

		train_split_data = split [0: -nb_of_test_data_by_split]

		if len (train_data) == 0:
			train_data = train_split_data
		else:
			train_data = np. concatenate ( (train_data, train_split_data), axis = 0)


		for i in range (len (models_params)):
			results = k_fold_cross_validation  (train_split_data, model = model, params = models_params [i], nb_sets = len (train_split_data) / n)
			for l in range (3):
			 	models_results[i][1][l] += results [l]

	# Get the best model parameters
	best_model, best_index = get_max_of_list (models_results)

	# Evaluate the best model on test data
	pred_model = train_model (train_data, model, models_params [best_index])
	score = test_model (test_data[: , 1:], test_data[: , 0], pred_model)

	return best_model[0], score

#==============================================================#

def  train_model (data, model, params):

	if model == "RF":
		pred_model = RandomForestClassifier(**params)

	elif model == "SVM":
		pred_model = svm. SVC (**params)

	elif model in ["RIDGE", "Ridge"]:
		pred_model = linear_model.Ridge (**params)

	elif model in ["LASSO", "Lasso"]:
		pred_model = linear_model.Lasso (**params)

	pred_model. fit (data[:,1:], data[:,0])

	return pred_model

#==============================================================#

def  test_model (X, Y, model):

	pred = model. predict (X)
	real = Y

	if type(model).__name__ in ["Ridge", "RIDGE", "Lasso", "LinearRegression"]:
		pred = discretize (pred)
		real = discretize (real)

	score = [recall_score (real, pred, average = 'weighted'), precision_score (real, pred, average = 'weighted'), f1_score (real, pred, average = 'weighted')]

	return score
