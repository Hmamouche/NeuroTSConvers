# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche
import sys
import os
import argparse
from tools import toSuppervisedData, write_line, list_convers
from training import *

import warnings
warnings.filterwarnings("ignore")

#============================================================

def process_multiple_subsets (set_of_behavioral_predictors, convers, target_column,subject, predictors, behavioral_predictors_,lag_max, k, add_target, model, filename, colnames, write_scores):

	for behavioral_predictors in set_of_behavioral_predictors:

		all_data = concat (subject, target_column, convers, lag_max, behavioral_predictors, predictors, add_target)

		# l split, for each split, chose k set as a test data
		best_model_params, score = k_l_fold_cross_validation (all_data, target_column, model, n = 45, k = 2, l = 1)

		if write_scores:
			row = [subject, target_column,'', best_model_params, str (behavioral_predictors)] + score
			write_line (filename, row, colnames, mode="a+")

#============================================================

def concat (subject, target_column, convers, lag, behavioral_predictors, predictors, add_target):
	data = pd. DataFrame ()

	for conv in convers:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)
		target = pd.read_pickle (filename)[target_column]
		# Load neurophysio and behavioral predictors
		if len (behavioral_predictors) > 0:
			external_data = get_behavioral_data (subject, conv, behavioral_predictors)
		else:
			external_data = None
		if len (predictors) > 0:
			physio = pd.read_pickle (filename)[predictors]
		else:
			physio = None

		# concatenate data of all conversations
		if data.shape [0] == 0:
			data = pd. concat ([target, physio, external_data], axis = 1)
			supervised_data = toSuppervisedData (data.values, lag, add_target = add_target, nb_inter = 0)
			data = np.concatenate ((supervised_data. targets[:,0:1], supervised_data.data), axis = 1)
		else:
			U = pd. concat ([target, physio, external_data], axis = 1)
			supervised_U = toSuppervisedData (U.values, lag, add_target = add_target, nb_inter = 0)
			U = np.concatenate ((supervised_U. targets[:,0:1], supervised_U.data), axis = 1)
			data =  np. concatenate ((data, U), axis = 0)

	return data

#====================================================================#

def predict_regions (regions, convers, subject, lag_max, k, model, model_type, filename, colnames, write_scores):

	for target_column in regions:
		print ("------", target_column, '------')
		num_region = int (target_column. split ('_')[-1])

		# Find the best parameters of the model using the cross-validation
		if model_type == "multiv":
			add_target = False

			behavioral_predictors_ = []

			#=============================================================

			if target_column in ["region_1"]:
				predictors = []
				behavioral_predictors_ = [
											{"colors_ts": ["colorfulness", "ratio"], "eyetracking_ts": ["Face", "Mouth", "Eyes"], "facial_features_ts": [" gaze_angle_x", " gaze_angle_y", " pose_Rx", " pose_Ry", " pose_Rz"]},
											{"colors_ts": ["colorfulness", "ratio"], "eyetracking_ts": ["Face"], "facial_features_ts": [" pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz"]},
											{"colors_ts": ["colorfulness", "ratio"], "eyetracking_ts": ["Face", "Eyes"], "facial_features_ts": [" pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz"]},
											{"colors_ts": ["colorfulness", "ratio"], "eyetracking_ts": ["Face"]},
											{"colors_ts": ["colorfulness", "ratio"], "facial_features_ts": [" pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz"]},
											{"colors_ts": ["colorfulness", "ratio"], "facial_features_ts":[" pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz", " AU10_r"," AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"],
											"eyetracking_ts": ["Face", "Mouth", "Eyes"]}
											]

			#=============================================================

			if target_column in ["region_2", "region_3"]:
				predictors = []
				behavioral_predictors_ = [{"speech_left_ts": ["IPU"]}, {"speech_ts": ["IPU"]}, {"speech_left_ts": ["IPU"], "speech_ts": ["IPU"]},
										{"speech_left_ts": ["talk"]}, {"speech_ts": ["talk"]}, {"speech_left_ts": ["talk"], "speech_ts": ["talk"]},
										{"speech_left_ts": ["IPU", "talk"]}, {"speech_ts": ["IPU", "talk"]}, {"speech_left_ts": ["IPU", "talk"], "speech_ts": ["IPU", "talk"]}]

			#=============================================================

			elif target_column in ["region_4", "region_5"]:
				predictors = []
				behavioral_predictors_ = [{"speech_left_ts": ["IPU"]}, {"speech_ts": ["IPU"]}, {"speech_left_ts": ["IPU"], "speech_ts": ["IPU"]},

										{"speech_left_ts": ["talk"]}, {"speech_ts": ["talk"]}, {"speech_left_ts": ["talk"], "speech_ts": ["talk"]},

										{"speech_left_ts": ["IPU", "talk"]}, {"speech_ts": ["IPU", "talk"]}, {"speech_left_ts": ["IPU", "talk"], "speech_ts": ["IPU", "talk"]},

										{"speech_left_ts": ["IPU", "talk", "Overlap"]}, {"speech_ts": ["IPU", "talk",  "Overlap"]}, {"speech_left_ts": ["IPU", "talk",  "Overlap"],
										"speech_ts": ["IPU", "talk"]}]

			#=============================================================

			elif target_column in ["region_6", "region_7"]:
				redictors = []
				behavioral_predictors_ = {"facial_features_ts": [" gaze_angle_x", " gaze_angle_y"]}

			elif target_column in ["region_8", "region_9"]:
				predictors = [target_column]
				behavioral_predictors_ = [{"facial_features_ts": [" gaze_angle_x", " gaze_angle_y", " pose_Rx", " pose_Ry", " pose_Rz"]},
				{"speech_ts": ["IPU", "FilledBreaks", "Feedbacks", "Discourses", "Particles"]},
				{"emotions_ts":['angry', 'disgust', 'fear', 'happy','sad', 'surprise', 'neutral']}]

			process_multiple_subsets (behavioral_predictors_, convers, target_column,subject, predictors,
									  behavioral_predictors_,lag_max, k, add_target, model, filename, colnames, write_scores)

#====================================================================#

def predict_subject (subject, regions, lag, k, model, model_type, write, remove):
	if not os. path. exists ("separate_results/sub-%02d"%subject):
		os. makedirs ("separate_results/sub-%02d"%subject)

	if not os. path. exists ("separate_results/sub-%02d/"%subject):
		os. makedirs ("separate_results/sub-%02d/"%subject)

	if model_type == 'multiv':
		filename_hh = "separate_results/sub-%02d/multivariate_%s_HH.csv"%(subject, model)
		filename_hr = "separate_results/sub-%02d/multivariate_%s_HR.csv"%(subject, model)

	# set output filenames
	else:
		filename_hh = "separate_results/sub-%02d/univariate_%s_HH.csv"%(subject, model)
		filename_hr = "separate_results/sub-%02d/univariate_%s_HR.csv"%(subject, model)
		behavioral_predictors = [None]

	# remove previous output files
	if remove:
		os. system ("rm %s"%filename_hh)
		os. system ("rm %s"%filename_hr)

	_regions = ["region_%d"%i for i in regions]
	subject = "sub-%02d"%subject
	target_index = 0 # column to predict must be the first column

	# For now, we expect 24 conversations per subject
	convers = list_convers ()
	# fill conversations HH and HR
	hh_convers = []
	hr_convers = []

	colnames = ["subject", "Region", "Type", "Parameters", "Predictors", "recall", "precision", "f1_score"]

	if len (convers) < 24:
		print ("Error, 24 conversations are required for each subject")
		exit (1)

	for i in range (len (convers)):
		if i % 2 == 0:
			hr_convers. append (convers[i])
		else:
			hh_convers. append (convers[i])

	# Predict HH conversations
	predict_regions (_regions, hh_convers, subject,
					 lag_max = lag, k = 2, model = model,
					 model_type = model_type, filename = filename_hh, colnames = colnames, write_scores = write)

	# Predict HR conversations
	predict_regions (_regions, hr_convers, subject,
					lag_max = lag, k = 2, model = model,
					model_type = model_type, filename = filename_hr, colnames = colnames, write_scores = write)
