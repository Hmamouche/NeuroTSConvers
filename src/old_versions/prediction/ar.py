# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os

import argparse

from tools import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#------------------------------------------
if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	parser.add_argument("--lag", "-p", default=1, type=int)
	parser.add_argument("--write", "-w", help="write results", action="store_true")
	args = parser.parse_args()
	
	if not os. path. exists ("results/sub-%02d"%args.subject):
		os. makedirs ("results/sub-%02d"%args.subject)
		
	filename_hh = "results/sub-%02d/predictions_ar_HH.txt"%args.subject
	filename_hr = "results/sub-%02d/predictions_ar_HR.txt"%args.subject
	
	os. system ("rm %s"%filename_hh)
	os. system ("rm %s"%filename_hr)
	#print (args)
	
	regions = ["region_73", "region_74", "region_75", "region_76", "region_79","region_80","region_87","region_88", "region_121", 
	"region_122", "region_123", "region_124"]
	
	# set parameters
	subject = "sub-%02d"%args.subject
	lag = args. lag
	target_index = 0
	convers = list_convers ()
	colnames = ["subject", "Region", "Type", "Predictors", "Lag", "Rmse"]
	
	for target_column in regions:
	
		# specifications
		min_result = []
		for lag in range (1, args.lag + 1):
			model = train_global_model (convers[0:-2], target_column, target_index, subject, [], None, lag, epochs=20, batch_size=1, verbose=0)
			results = test_global_model (model, convers[-2:], target_index, target_column, subject, [], None, lag, epochs=1, batch_size=1, verbose=0)
			
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




