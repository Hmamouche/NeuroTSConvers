# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse
from tools import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
		
	filename_hh = "results/sub-%02d/disc/predictions_HH.txt"%args.subject
	filename_hr = "results/sub-%02d/disc/predictions_HR.txt"%args.subject
	
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
					model = train_global_model (convers[i:i+4], target_column, target_index, subject, predictors, external_predictors, lag, epochs=500, add_target = True)
					
					for conv in convers[i + 4 : i + 6]:
						score = test_global_model (model, conv, target_index, target_column, subject, predictors, external_predictors, lag, epochs=10, add_target = True)
									
						if int (conv. split ('_')[-1]) % 2 == 1:
							score_hh. append (score)

						else:
							score_hr. append (score)
					
				results = [np. mean (score_hh), np. mean (score_hr)]
				
				if lag == 1:
					min_result = results[:]
					best_lag = lag

				else:					
					if (np. mean (results) < np. mean (min_result)):
						min_result = results[:]
						best_lag = lag
				
			if args.write:
				row_hh = [subject, target_column,"HH", variables, best_lag] + min_result[0]
				row_hr = [subject, target_column,"HR", variables, best_lag] + min_result[1]
				write_line (filename_hh, row_hh, colnames, mode="a+")
				write_line (filename_hr, row_hr, colnames, mode="a+")

		
		



