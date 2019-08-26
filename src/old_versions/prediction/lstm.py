# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse
from tools import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#--------------------------------------------------------

def build_lstm (data, in_dim, out_dim, nb_epochs, set_coefs = True):
	# define our MLP network
	model = Sequential()
	model.add (LSTM (neurons, batch_input_shape = (batch_size, X.shape[1], X.shape[2])))
	model.add(Dense(1))
	
	model.compile (loss='mean_squared_error', optimizer='adam')
    
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	
	return model
	
#--------------------------------------------------------

def fit_lstm (data, target_index, lag, model, normalize = False, epochs=10, batch_size=1, verbose=0) :

	# normalize data
	if normalize:
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)
		
	supervised_data = toSuppervisedData (data, lag)
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]
 	X = X.reshape(X.shape[0], lag, X.shape[1] / look_back)
 	model.reset_states()
	 
	model.fit (X, Y, epochs = epochs, batch_size = batch_size, verbose = verbose)
	
#--------------------------------------------------------

#----------------------------------------------------------------------------------------------------				
def  train_global_lstm (convers, target_column, target_index, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0):
	# convers: list of conversations file names
	# target_index
	# sunject
	# predictors: the physio variables names to add in the model
	# lag : the lag parameter
	# Read first filename
	
	filename = "time_series/%s/physio_ts/%s.pkl"%(subject, convers[0])
	
	if external_predictors != None:
		external_data = get_behavioral_data (subject, convers[0], external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]

	# normalize data
	scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data =  scaler. transform (data)
	
	# Build lstm model
	model = build_lstm (X.shape[1], 1, start_weigths = weights)

	# Online model updating on  the rest of conversations
	for conv in convers[1:]:
		filename = "time_series/%s/physio_ts/%s.pkl"%(subject, conv)
		
		if external_predictors != None:
			external_data = get_behavioral_data (subject, conv, external_predictors)
			data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
		else:
			pd.read_pickle (filename)[[target_column] + predictors]
			
		fit_lstm (data, target_index, lag, model, normalize = True, epochs = epochs, batch_size = batch_size, verbose=verbose)
		
	return model
	
if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	parser.add_argument("--lag", "-p", default=1, type=int)
	parser.add_argument("--write", "-w", help="write results", action="store_true")
	args = parser.parse_args()
	
	if not os. path. exists ("results/sub-%02d"%args.subject):
		os. makedirs ("results/sub-%02d"%args.subject)
		
	filename_hh = "results/sub-%02d/predictions_HH.txt"%args.subject
	filename_hr = "results/sub-%02d/predictions_HR.txt"%args.subject
	
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
	colnames = ["subject", "Region", "Type", "Predictors", "Lag", "Rmse"]
	
	for target_column in regions:
	
		# specifications
		if target_column in ["region_75", "region_76", "region_79","region_88", "region_121", "region_122", "region_123", "region_124"]:
			behavioral_predictors = [{"speech_ts": ["Silence","Overlap","reactionTime","filledBreaks"]},
								{"speech_ts": ["Silence"]}]
		
		elif target_column in ["region_73", "region_74"]:
			behavioral_predictors = [{"speech_ts": ["Signal-env"]}, {"speech_ts": ["Silence"]}, {"speech_ts": ["Silence", "Signal-env"]}]
			
		elif target_column in ["region_87", "region_88"]:
			behavioral_predictors = [{"speech_ts": ["Silence"]}, {"speech_ts": ["Silence", "Signal-env"]}]

		for  external_predictors in  behavioral_predictors:
			min_result = []
			for lag in range (1, args.lag + 1):
				model = train_global_lstm (convers[0:-2], target_column, target_index, subject, predictors, external_predictors, lag, epochs=20, batch_size=1, verbose=0)
				results = test_global_model (model, convers[-2:], target_index, target_column, subject, predictors, external_predictors, lag, epochs=1, batch_size=1, verbose=0)
					
				if lag == 1:
					min_result = results[:]

				else:					
					if (results[0][-1] + results[1][-1]) < (float (min_result[0][-1]) + float (min_result[1][-1])):
						min_result = results[:]
						
			if args.write:
				if min_result[0][2] == "HH":
					write_line (filename_hh, min_result[0], colnames, mode="a+")
					write_line (filename_hr, min_result[1], colnames, mode="a+")
				else:
					write_line (filename_hr, min_result[0], colnames, mode="a+")
					write_line (filename_hh, min_result[1], colnames, mode="a+")
		
		



