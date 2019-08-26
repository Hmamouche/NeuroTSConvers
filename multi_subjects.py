import os
from joblib import Parallel, delayed
import argparse

import sys
sys.path.append ("src/prediction")
sys.path.append ("src/")

from prediction import predict_all
from feature_selection import select_features

import random

def prediction (subjects, regions, lag, blocks, remove, lstm):

	if lstm:
		predict_all (subjects, regions, lag, blocks, "LSTM", remove)

	else:
		predict_all (subjects, regions, lag, blocks, "GB", remove)
		#predict_subject (subjects, regions, lag, blocks, "SGD", "multiv", remove)
		predict_all (subjects, regions, lag, blocks, "SVM", remove)
		predict_all (subjects, regions, lag, blocks, "RF", remove)
		predict_all (subjects, regions, lag, blocks, "random", remove)

	print ("... Done.")

if __name__=='__main__':

	# read arguments
	parser = argparse. ArgumentParser ()
	parser. add_argument ('--subjects', '-s', nargs = '+', type=int)
	parser. add_argument ('--type', '-t', help = "typx = e of the task", default = "prediction")
	parser. add_argument ("--lag", "-p", default = 5, type=int)
	parser. add_argument ("--blocks", "-b", help = "number of split in k_fold_cross_validation", default=2, type=int)
	parser. add_argument ("--write", "-w", help = "write results", action="store_true")
	parser. add_argument ("--remove", "-rm", help = "remove previous files", action="store_true")
	parser. add_argument ("--lstm", "-lstm", help = "using lstm model", action="store_true")
	parser. add_argument ('--regions','-rg', nargs = '+', type=int)


	if not os. path. exists ("results/prediction"):
		os. makedirs ("results/prediction")

	if not os. path. exists ("results/selection"):
		os. makedirs ("results/selection")


	args = parser.parse_args()
	print (args)


	if args. type in  ["selection", "selec"]:
		select_features (args. subjects, args. regions, args. lag, args. remove)

	elif args. type in  ["prediction", "pred"]:
		prediction (args.subjects, args.regions, args.lag, args.blocks, args.remove, args.lstm)
