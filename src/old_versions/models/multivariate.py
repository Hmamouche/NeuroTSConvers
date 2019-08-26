# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse
from tools import toSuppervisedData, write_line, list_convers
from models_tools import *

import warnings
warnings.filterwarnings("ignore")

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
	colnames = ["subject", "Region", "Type", "Predictors", "Lag", "recall", "precision", "f1_score"]

	if len (convers) < 24:
		print ("Error, 24 conversations are required for each subject")
		exit (1)

	prediction_type = ""
	print ("prediction type \n", prediction_type)
	for target_column in regions:

		print ("------", target_column, '------')

		num_region = int (target_column. split ('_')[-1])

		predictors = []

		behavioral_predictors = [{"speech_ts": ["Silence", "Signal-env", "Overlap","reactionTime","filledBreaks"]},
								{"speech_ts": ["Silence", "Signal-env", "Overlap","filledBreaks"]},
							    {"speech_ts": ["Signal-env"]},
								{"speech_ts": ["Silence"]},
								{"speech_ts": ["Silence", "Signal-env", "lex1", "lex2"]},
								{"speech_ts": ["Silence", "Signal-env"]}]


		behavioral_predictors =  [
									{"speech_ts": ["Silence", "Signal-env"],
									"facial_features_ts": [" AU01_r"," AU02_r"," AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU09_r", " AU10_r",
															" AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"," AU45_r"]},

									{"speech_ts": ["Silence", "Signal-env"]},

									{"speech_ts": ["Silence", "Signal-env"],
									"facial_features_ts": [" gaze_angle_x", " gaze_angle_y", , " pose_Rx", " pose_Ry", " pose_Rz",
									" AU10_r"," AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"]},

									{"speech_ts": ["Silence", "Signal-env"],
									"facial_features_ts": [" gaze_angle_x", " gaze_angle_y", " pose_Rx", " pose_Ry", " pose_Rz"]}

								 ]

		# Find the best parameters of the model using the cross-validation
		best_model_params, score = kfold_cross_validation (convers, target_column, target_index,
																				  subject, predictors, behavioral_predictors,
																				  lag_max = args. lag, k = 3, add_target = True, model = args.model)


		best_variables = get_items ([], best_model_params [0])

		if args. write:
			row_hh = [subject, target_column,"HH", best_variables, best_model_params[1]] + score [0]. tolist ()
			row_hr = [subject, target_column,"HR", best_variables, best_model_params[1]] + score [1]. tolist ()
			write_line (filename_hh, row_hh, colnames, mode="a+")
			write_line (filename_hr, row_hr, colnames, mode="a+")
