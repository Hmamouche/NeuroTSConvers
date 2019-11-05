import os
from joblib import Parallel, delayed
import argparse
import pandas as pd

import sys
sys.path.append ("src/prediction")
sys.path.append ("src/")

#from prediction_by_subj import predict_all
#from run import predict_all
from prediction import predict_all
from src. feature_selection. feature_selection import select_features

#import random

def prediction (subjects, regions, lag, blocks, remove, lstm, find_params):

	if lstm:
		predict_all (subjects, regions, lag, blocks, "LSTM", remove, find_params)

	else:
		#predict_all (subjects, regions, lag, blocks, "GB", remove, find_params)
		#predict_all (subjects, regions, lag, blocks, "SGD", "multiv", remove, find_params)
		#predict_all (subjects, regions, lag, blocks, "SVM", remove, find_params)
		predict_all (subjects, regions, lag, blocks, "RF", remove, find_params)
		#predict_all (subjects, regions, lag, blocks, "MLP", remove, find_params)
		#predict_all (subjects, regions, lag, blocks, "FUZZY", remove, find_params)
		#predict_all (subjects, regions, lag, blocks, "BAG", remove, find_params)

		predict_all (subjects, regions, lag, blocks, "baseline", remove, find_params)

		#predict_all (subjects, regions, lag, blocks, "RIDGE", remove, find_params)

	print ("... Done.")

if __name__=='__main__':

	# read arguments
	parser = argparse. ArgumentParser ()
	parser. add_argument ('--subjects', '-s', nargs = '+', type=int)
	parser. add_argument ('--type', '-t', help = "type = e of the task", default = "prediction")
	parser. add_argument ("--lag", "-p", default = 5, type=int)
	parser. add_argument ("--blocks", "-b", help = "number of split in k_fold_cross_validation", default=2, type=int)
	parser. add_argument ("--write", "-w", help = "write results", action="store_true")
	parser. add_argument ("--remove", "-rm", help = "remove previous files", action="store_true")
	parser. add_argument ("--lstm", "-lstm", help = "using lstm model", action="store_true")
	parser. add_argument ('--regions','-rg', nargs = '+', type=int)
	parser. add_argument ("--find_params", "-crossv", help = "find the parameters of the models with cross validation", action="store_true")

	args = parser.parse_args()

	brain_areas_desc = pd. read_csv ("brain_areas.tsv", sep = '\t', header = 0)
	brain_areas = []

	for num_region in args. regions:
		brain_areas. append (brain_areas_desc . loc [brain_areas_desc ["Code"] == num_region, "Name"]. values [0])


	if not os. path. exists ("results/prediction"):
		os. makedirs ("results/prediction")

	if not os. path. exists ("results/models_params"):
		os. makedirs ("results/models_params")

	if not os. path. exists ("results/selection"):
		os. makedirs ("results/selection")


	print (args)

	if args. type in  ["selection", "selec"]:
		select_features (args. subjects, brain_areas, args. lag, args. remove)

	elif args. type in  ["prediction", "pred"]:
		prediction (args.subjects, brain_areas, args.lag, args.blocks, args.remove, args.lstm, args. find_params)
