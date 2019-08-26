# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse
import random as rd

from tools import toSuppervisedData, write_line, list_convers
from models_tools import *

import warnings
warnings.filterwarnings("ignore")

if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	parser.add_argument("--lag", "-p", default=1, type=int)
	parser.add_argument("--write", "-w", help="write results", action="store_true")
	args = parser.parse_args()

	if not os. path. exists ("results/sub-%02d"%args.subject):
		os. makedirs ("results/sub-%02d"%args.subject)

	#if not os. path. exists ("results/sub-%02d/disc_univariate"%args.subject):
		#os. makedirs ("results/sub-%02d/disc_univariate"%args.subject)

	filename_hh = "results/sub-%02d/univariate_RF_HH.csv"%args.subject
	filename_hr = "results/sub-%02d/univariate_RF_HR.csv"%args.subject

	os. system ("rm %s"%filename_hh)
	os. system ("rm %s"%filename_hr)

	regions = ["region_73", "region_74", "region_75","region_76", "region_79","region_80",
			 "region_87","region_88", "region_121","region_122", "region_123", "region_124"]

	# set parameters
	subject = "sub-%02d"%args.subject
	lag = args. lag
	target_index = 0
	predictors = []
	convers = list_convers ()
	colnames = ["subject", "Region", "Type", "Predictors", "Lag", "recall", "precision", "f1_score"]

	if len (convers) < 24:
		print ("Error, 24 conversations are required for each subject")
		exit (1)


	for target_column in regions:

		print ("------", target_column, '------')

		num_region = int (target_column. split ('_')[-1])
		predictors = []
		behavioral_predictors = [None]

		# Find the best parameters of the model using the cross-validation
		best_model_params, score = kfold_cross_validation (convers, target_column, target_index,
																				  subject, predictors, behavioral_predictors,
																				  lag_max = args. lag, k = 2, add_target = True)

		#print (best_model_params, score)

		if args. write:
			row_hh = [subject, target_column,"HH", "", best_model_params[1]] + score [0]. tolist ()
			row_hr = [subject, target_column,"HR", "", best_model_params[1]] + score [1]. tolist ()
			write_line (filename_hh, row_hh, colnames, mode="a+")
			write_line (filename_hr, row_hr, colnames, mode="a+")
