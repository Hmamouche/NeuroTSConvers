# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse
from tools import toSuppervisedData, write_line, list_convers
from models_tools import *

import warnings
warnings.filterwarnings('always')

#============================================================

def concat (convers, lag, behavioral_predictors, predictors):
	data = pd. DataFrame ()
	for conv in convers:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		if behavioral_predictors != None:
			external_data = get_behavioral_data (subject, conv, behavioral_predictors)
			if data.shape [0] == 0:
				data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1)
			else:
				U = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1)
				data =  pd. concat ( [data, U], axis = 0)

		else:
			if data.shape [0] == 0:
				data = pd.read_pickle (filename)[[target_column] + predictors]
			else:
				U =  pd.read_pickle (filename)[[target_column] + predictors]
				data =  pd. concat ( [data, U], axis = 0)

	# normalize data
	scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data =  scaler. transform (data)

	return data, scaler

#============================================================

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	parser.add_argument("--lag", "-p", default=1, type=int)
	parser.add_argument("--write", "-w", help="write results", action="store_true")
	parser.add_argument("--model", "-m",  help="Prediction model",  choices=["RF", "SVM"])
	args = parser.parse_args()

	if not os. path. exists ("results/sub-%02d"%args.subject):
		os. makedirs ("results/sub-%02d"%args.subject)

	if not os. path. exists ("results/sub-%02d/"%args.subject):
		os. makedirs ("results/sub-%02d/"%args.subject)

	filename_hh = "results/sub-%02d/multivariate_%s_HH.csv"%(args.subject, args. model)
	filename_hr = "results/sub-%02d/multivariate_%s_HR.csv"%(args.subject, args. model)

	os. system ("rm %s"%filename_hh)
	os. system ("rm %s"%filename_hr)

	regions = ["region_73", "region_74", "region_75","region_76", "region_79","region_80",
			 "region_87","region_88", "region_121","region_122", "region_123", "region_124"]

	# set parameters
	subject = "sub-%02d"%args.subject
	lag = args. lag
	target_index = 0

	convers = list_convers ()



	#-----------  Prediction
	for target_column in regions:

		print ("------", target_column, '------')
		num_region = int (target_column. split ('_')[-1])
		predictors = [target_column]
		behavioral_predictors =  {"speech_ts": ["Silence", "Signal-env"]}




		print (data)

		# split data into k sets
		k =  4



		exit (1)
