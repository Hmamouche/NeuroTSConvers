# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse
import random as rd
from tools import toSuppervisedData, write_line, list_convers
from randf_tools import *

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	parser.add_argument("--lag", "-p", default=1, type=int)
	parser.add_argument("--write", "-w", help="write results", action="store_true")
	parser.add_argument("--pred_type", "-t",  help="Discretize predictors",  action="store_true")
	args = parser.parse_args()


	if not os. path. exists ("results/sub-%02d"%args.subject):
		os. makedirs ("results/sub-%02d"%args.subject)

	if args.pred_type:
		prediction_type = 'discretized_'
		if not os. path. exists ("results/sub-%02d/disc_all"%args.subject):
			os. makedirs ("results/sub-%02d/disc_all"%args.subject)

		filename_hh = "results/sub-%02d/disc_all/predictions_randF_HH.txt"%args.subject
		filename_hr = "results/sub-%02d/disc_all/predictions_randF_HR.txt"%args.subject

	else:
		prediction_type = ""
		if not os. path. exists ("results/sub-%02d/disc"%args.subject):
			os. makedirs ("results/sub-%02d/disc"%args.subject)

		filename_hh = "results/sub-%02d/disc/predictions_randF_HH.txt"%args.subject
		filename_hr = "results/sub-%02d/disc/predictions_randF_HR.txt"%args.subject

	os. system ("rm %s"%filename_hh)
	os. system ("rm %s"%filename_hr)

	regions = ["region_73", "region_74", "region_75","region_76", "region_79","region_80",
			 "region_87","region_88", "region_121","region_122", "region_123", "region_124"]

	# set parameters
	subject = "sub-%02d"%args.subject
	lag = args. lag
	target_index = 0

	convers = list_convers ()
	colnames = ["subject", "Region", "Type", "Predictors", "Lag", "recall", "precision", "f1_score"]

	if len (convers) < 24:
		print ("Error, 24 conversations are required for each subject")
		exit (1)


	convers_test = convers[-2:]
	convers = convers [0:-2]


	print ("prediction type \n", prediction_type)
	for target_column in regions:

		num_region = int (target_column. split ('_')[-1])

		if  (num_region % 2) == 1:
			predictors = ["region_%d" %(num_region + 1)]
		else:
			predictors = ["region_%d" %(num_region - 1)]

		behavioral_predictors = [{prediction_type + "speech_ts": ["Silence", "Signal-env", "Overlap","reactionTime","filledBreaks"]},
								{prediction_type + "speech_ts": ["Silence", "Overlap","reactionTime","filledBreaks"]},
							    {prediction_type + "speech_ts": ["Signal-env"]},
								{prediction_type + "speech_ts": ["Silence"]},
								{prediction_type + "speech_ts": ["Silence", "Signal-env"]}]

		best_set_result = []
		best_predictors = []
		best_variables = []
		for  external_predictors in  behavioral_predictors:

			if external_predictors != None:
				external_variables = []
				for key in external_predictors. keys ():
					external_variables += external_predictors[key]
				variables = '+'.join (predictors + external_variables)
			else:
				variables = '+'.join (predictors)

			best_result = []
			best_lag = 1

			for lag in range (1, args.lag + 1):
				score_hh = []
				score_hr = []
				# cross_validation over 22 conversations
				# First, rearange conversations randomly
				convers_order = rd. sample (range (22), k = 22)

				# testing the model in on cobversations each time, and train it oh the others
				i = 0
				while (i < len (convers_order)):
					test_convers = [convers [convers_order [i]], convers [convers_order[i + 1]]]
					train_convers = [convers [j] for j in convers_order]

					train_convers. remove (test_convers[0])
					train_convers. remove (test_convers[1])

					model = train_random_forrest (train_convers, target_column, target_index, subject, predictors, external_predictors, lag, add_target = True)

					for conv in test_convers:
						score = test_random_forest (model, conv, target_index, target_column, subject, predictors, external_predictors, lag, add_target = True)

						if int (conv. split ('_')[-1]) % 2 == 1:
							score_hh. append (score)

						else:
							score_hr. append (score)

					i += 2

				results = [np. mean (score_hh, axis = 0), np. mean (score_hr, axis = 0)]

				if lag == 1:
					best_result = results[:]

				else:
					if (results[0][2] + results[1][2]) > np. mean (best_result[0][2] + best_result[1][2]):
						best_result = results[:]
						best_lag = lag

			if len (best_set_result) == 0 or ((best_result[0][2] + best_result[1][2]) > np. mean (best_set_result[0][2] + best_set_result[1][2])):
				best_set_result = best_result[:]
				best_variables = variables
				best_set_lag = best_lag
				best_predictors = external_predictors


		if args.write:
			model = train_random_forrest (convers, target_column, target_index, subject, predictors, best_predictors, best_set_lag, add_target = True)
			score_hh = []
			score_hr = []

			for conv in convers_test:
				score = test_random_forest (model, conv, target_index, target_column, subject, predictors, best_predictors, best_set_lag, add_target = True)

				if int (conv. split ('_')[-1]) % 2 == 1:
					score_hh = (score)

				else:
					score_hr = (score)

			row_hh = [subject, target_column,"HH", best_variables, best_lag] + score_hh
			row_hr = [subject, target_column,"HR", best_variables, best_lag] + score_hr
			write_line (filename_hh, row_hh, colnames, mode="a+")
			write_line (filename_hr, row_hr, colnames, mode="a+")
