import sys
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

from tools import toSuppervisedData, get_behavioral_data

#----- Fit and train random forrest
def  train_random_forrest (convers, target_column, target_index, subject, predictors, external_predictors, lag, add_target = True):

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
	clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
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
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]

	pred = model. predict (X)
	real = Y

	score = [recall_score (real, pred), precision_score (real, pred), f1_score (real, pred)]

	return score
