# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse

from keras.layers import LSTM

from tools import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--------------------------------------------------------

def build_lstm (data, target_index, lag, nb_epochs, normalize = True, batch_size = 1, add_target = True):

	# normalize data
	if normalize:
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)

	supervised_data = toSuppervisedData (data, lag, add_target = add_target)
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]
	X = X.reshape(X.shape[0], int (lag), int (X.shape[1] / lag))

	# define our MLP network
	model = Sequential()
	model.add (LSTM (lag, batch_input_shape = (batch_size, X.shape[1], X.shape[2])))
	model.add (Dense (1))
	model. add (Activation ("sigmoid"))

	#model.compile (loss='mean_squared_error', optimizer='adam')
	model. compile (optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

	for i in range(nb_epochs):
		model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()

	return model

#--------------------------------------------------------

def fit_lstm (data, target_index, lag, model, normalize = False, epochs=10, batch_size=1, verbose=0, add_target = True) :

	# normalize data
	if normalize:
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)

	supervised_data = toSuppervisedData (data, lag, add_target = add_target)
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]
	X = X.reshape(X.shape[0], int (lag), int (X.shape[1] / lag))
	model.reset_states()

	model.fit (X, Y, epochs = epochs, batch_size = batch_size, verbose = verbose)

#----------------------------------------------------------------------------------------------------
def  train_global_lstm (convers, target_column, target_index, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0, add_target = True):
	# convers: list of conversations file names
	# target_index
	# subject : subject name
	# predictors: the physio variables names to add in the model
	# lag : the lag parameter
	# Read first filename

	filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, convers[0])

	if external_predictors != None:
		external_data = get_behavioral_data (subject, convers[0], external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]


	# Build lstm model
	model = build_lstm (data, target_index, lag, epochs, add_target = add_target)

	# Online model updating on  the rest of conversations
	for conv in convers[1:]:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		if external_predictors != None:
			external_data = get_behavioral_data (subject, conv, external_predictors)
			data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
		else:
			data = pd.read_pickle (filename)[[target_column] + predictors]

		fit_lstm (data, target_index, lag, model, normalize = True, epochs = epochs, batch_size = batch_size, verbose=verbose, add_target = add_target)

	return model

#----------------------------------------------------------------------------------------------------
def  test_global_lstm (model, conv, target_index, target_column, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0, lag_max=5, add_target = True):
	results = []

	filename = "time_series/%s/discretized_physio_ts/%s.pkl" %(subject, conv)

	if not os.path.exists (filename):
		print ("file does not exist")

	if external_predictors != None:
		external_data = get_behavioral_data (subject, conv, external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]

	scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data = scaler. transform (data)

	real = data [lag_max:,target_index]
	start_points = data [0:lag,:]
	pred = []

	for i in range (lag_max, data. shape [0]):

		supervised_data = toSuppervisedData (data[i-lag:i+1,:], lag, add_target = add_target)
		X = supervised_data.data
		Y = supervised_data.targets [:,target_index]
		X = X.reshape(X.shape[0], int (lag), int (X.shape[1] / lag))

		# Make one prediction
		#predictions = predict_var_mlp (data[i-lag-1:i,:], target_index, lag, model, normalize = False, batch_size=1)
		predictions = model. predict (X, batch_size = batch_size)

		if (predictions[-1][0] > 0.4):
			pred. append (1)
		else:
			pred. append (0)

		# Update the model
		model. fit (X, Y, verbose = verbose, epochs=epochs, batch_size = batch_size)

	score = [accuracy_score (real, pred), precision_score (real, pred), f1_score (real, pred)]

	return score

#-----------------------------------------------
if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	parser.add_argument("--lag", "-p", default=1, type=int)
	parser.add_argument("--write", "-w", help="write results", action="store_true")
	args = parser.parse_args()

	if not os. path. exists ("results/sub-%02d"%args.subject):
		os. makedirs ("results/sub-%02d"%args.subject)

	if not os. path. exists ("results/sub-%02d/disc"%args.subject):
		os. makedirs ("results/sub-%02d/disc"%args.subject)

	filename_hh = "results/sub-%02d/disc/predictions_lstm_HH.txt"%args.subject
	filename_hr = "results/sub-%02d/disc/predictions_lstm_HR.txt"%args.subject

	os. system ("rm %s"%filename_hh)
	os. system ("rm %s"%filename_hr)

	regions = ["region_73", "region_74", "region_75",
			 "region_76", "region_79","region_80",
			 "region_87","region_88", "region_121",
			 "region_122", "region_123", "region_124"]

	# set parameters
	subject = "sub-%02d"%args.subject
	lag = args. lag
	target_index = 0
	predictors = []
	convers = list_convers ()
	colnames = ["subject", "Region", "Type", "Predictors", "Lag", "accuracy", "precision", "fscore"]

	for target_column in regions:


		behavioral_predictors = [None, {"speech_ts": ["Silence", "Signal-env", "Overlap","reactionTime","filledBreaks"]},
				{"speech_ts": ["Signal-env"]}, {"speech_ts": ["Silence"]}, {"speech_ts": ["Silence", "Signal-env"]}]


		# specifications
		'''if target_column in ["region_75", "region_76", "region_79","region_88", "region_121", "region_122", "region_123", "region_124"]:
			behavioral_predictors = [{"speech_ts": ["Silence","Overlap","reactionTime","filledBreaks"]},
								{"speech_ts": ["Silence"]}]

		elif target_column in ["region_73", "region_74"]:
			behavioral_predictors = [{"speech_ts": ["Signal-env"]}, {"speech_ts": ["Silence"]}, {"speech_ts": ["Silence", "Signal-env"]}]

		elif target_column in ["region_87", "region_88"]:
			behavioral_predictors = [{"speech_ts": ["Silence"]}, {"speech_ts": ["Silence", "Signal-env"]}]'''

		for  external_predictors in  behavioral_predictors:

			# Join all predictors by ';'
			if external_predictors != None:
				external_variables = []
				for key in external_predictors. keys ():
					external_variables += external_predictors[key]
				variables = '+'.join (predictors + external_variables)
			else:
				variables = '+'.join (predictors)


			min_result = []
			global_results = []
			score_hh = []
			score_hr = []

			for lag in range (1, args.lag + 1):
				if len (convers) < 24:
					print ("Error, 24 conversations are required for each subject")
					exit (1)

				for i in range (4):
					model = train_global_lstm (convers[i:i+4], target_column, target_index, subject, predictors, external_predictors, lag, epochs=200, add_target = True)

					for conv in convers[i+4:i+6]:
						score = test_global_lstm (model, conv, target_index, target_column, subject, predictors, external_predictors, lag, epochs=10, add_target = True)

						if int (conv. split ('_')[-1]) % 2 == 1:
							score_hh. append (score)

						else:
							score_hr. append (score)

				results = [np. mean (score_hh, axis = 0), np. mean (score_hr, axis = 0)]

				if lag == 1:
					min_result = results[:]
					best_lag = lag

				else:
					if (np. mean (results[0][1:]) > np. mean (min_result[0][1:])):
						min_result = results[:]
						best_lag = lag

			#print (min_result[0])

			if args.write:
				row_hh = [subject, target_column,"HH", variables, best_lag] + min_result[0]. tolist ()
				row_hr = [subject, target_column,"HR", variables, best_lag] + min_result[1]. tolist ()
				write_line (filename_hh, row_hh, colnames, mode="a+")
				write_line (filename_hr, row_hr, colnames, mode="a+")

						#if int (conv. split ('_')[-1]) % 2 == 1:
							#score_hh. append (score)
