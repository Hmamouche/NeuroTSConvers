import numpy as np
import pandas as pd

from src. fcbf import fcbf
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from collections import defaultdict
from itertools import chain
#============================================================

def merge_two_dict (a, b):

	c = defaultdict (list)
	for k, v in chain (a.items (), b.items ()):
		c[k].extend(v)

	return c

#============================================================

def merge_dict (list_dict):
	if len (list_dict) == 1:
		return list_dict [0]

	c = list_dict [0]
	for i in range (1, len (list_dict)):
		c = merge_two_dict (c, list_dict[i])

	return c

#===============================================================#

def reduce (train_, test_, method, perc_comps):

	"""
	Reduce the dimmensionality of the train and test data with model fitted on train data
	method : dimension reduction or feature selection methode used
	percs_comps : percentage of n_components in case of DR methods, and a threshold for the FCBF method.
	"""

	# make a copy to do no change the input data
	train = train_.copy ()
	test = test_. copy ()

	if method == "None":
	    return train, test, range (0, train. shape[1] - 1)

	elif method == "RFE":
	    return rfe_reduction (train_, test_, perc_comps)

	if method == "FCBF":
	    sbest = fcbf (train [:, 1: ], train [:, 0], perc_comps)

	    if len (sbest) == 0:
	    	return train, test, range (0, train. shape[1] - 1)

	    best_indices = [int (a) for a in sbest[:, 1]]

	    train = train [:,  [0] + [int (a) + 1 for a in best_indices]]
	    test = test [:,  [0] + [int (a) + 1 for a in best_indices]]

	    return train, test, best_indices

	n_comps = (train. shape[1] - 1) * perc_comps
	n_comps = int (n_comps)

	if method == "PCA":
	    model = PCA (n_components = n_comps)

	elif method == "KPCA":
	    model = KernelPCA (n_components = n_comps)

	elif method == "IPCA":
	    model = IncrementalPCA (batch_size = None, n_components = n_comps)

	model = model.fit (train [:, 1: ])

	train = np.concatenate ((train [:, 0:1], model. transform (train [:, 1: ])), axis=1)
	test = np.concatenate ((test [:, 0:1], model. transform (test [:, 1: ])), axis=1)

	return train, test, range (0, train. shape[1] - 1)

#===============================================================#

def rfe_reduction (train_, test_, percs):

	"""
	Reduce train and test data with a model fitted on train data
	based on the recursive feature elimination method.
	percs: percentage of features to select
	"""

	score = 0
	results = []
	support = []

	# make a copy to do not change the input data
	train = train_.copy ()
	test = test_. copy ()

	# find the best subset of features
	for perc_comps in percs:

		# Compute the number of features to select
		k = int ((train_. shape[1] - 1) * perc_comps)

		if k <= 0:
		    continue

		# Prediction model to use
		estimator = RandomForestClassifier (n_estimators = 150, max_features = 'auto', bootstrap = True, max_depth = 10)
		selector = RFE (estimator, k, step = 1)
		selector = selector.fit (train [:, 1: ], train [:, 0])

		if score < selector. score (train [:, 1: ], train [:, 0]):
		    score = selector. score (train [:, 1: ], train [:, 0])
		    support = selector. support_

	best_indices = []

	if len (support) == 0:
		best_indices = [i for i in range (train. shape [1] - 1)]
		score = 1

	else:
		for i in range (len (support)):
		    if support[i]:
		        best_indices. append (i)

	return (train [:, [0] + best_indices], test [:, [0] + best_indices], best_indices, score)


#======================================================================================
def manual_selection (region):

    # TODO: store information in a file instead of computing each time dictionaries
	AUrs = {"facial_features_ts": [" AU01_r"," AU02_r"," AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU23_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"]}
	AUcs = {"facial_features_ts": [" AU01_c"," AU02_c"," AU04_c"," AU05_c"," AU06_c"," AU07_c"," AU23_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c"," AU20_c"," AU23_c"," AU25_c"," AU26_c"]}
	eyetracking =  {"face_ts": ["Face", "Mouth", "Eyes"]}
	face =  {"face_ts": ["Face"]}
	gradient = {"eyes_gradient_ts": ["Vx", "Vy"]}
	head_pose_r = {"facial_features_ts": [" pose_Rx", " pose_Ry", " pose_Rz"]}
	head_pose_t = {"facial_features_ts": [" pose_Tx", " pose_Ty", " pose_Tz"]}
	gaze_angle = {"facial_features_ts": [" gaze_angle_x", " gaze_angle_y"]}
	colors = {"colors_ts": ["colorfulness"]}

	audio = {"speech_ts": ["Signal"]}
	audio_left = {"speech_left_ts": ["Signal"]}
	speech_ipu = {"speech_ts": ["IPU"]}
	speech_left_ipu = {"speech_left_ts": ["IPU"]}
	speech_talk = {"speech_ts": ["talk"]}
	speech_left_talk = {"speech_left_ts": ["talk"]}

	all_0 = face
	all_1 = merge_dict ([AUcs, eyetracking, head_pose_r])
	all_2 = merge_dict ([eyetracking, gradient])
	all_3 = merge_dict ([face, gradient])
	all_4 = merge_dict ([eyetracking, head_pose_r, head_pose_t])
	all_5 = merge_dict ([face, gradient, head_pose_r, head_pose_t])
	all_6 = merge_dict ([head_pose_r, head_pose_t])
	all_7 = merge_dict ([face, gradient, head_pose_r, head_pose_t])
	all_8 = merge_dict ([eyetracking, AUrs])
	all_9 = merge_dict ([head_pose_r, head_pose_t, AUcs])
	all_10 = merge_dict ([face, head_pose_r, head_pose_t])


	items = {"speech_ts": ["FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters"]}
	items_left = {"speech_left_ts": ["FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters"]}
	speech_items = merge_dict ([speech_ipu, items])
	speech_items_left = merge_dict ([speech_left_ipu, items_left])

	if region in ["region_1"]:
		set_of_behavioral_predictors = [gradient, face, all_1, all_3, all_4, all_5, all_6, all_7, all_8, all_9, all_10]



	#elif region in ["region_2", "region_3"]:
		#set_of_behavioral_predictors = [speech_left_ipu, speech_ipu,
										#speech_talk, merge_dict ([speech_ipu, speech_talk]),  merge_dict ([speech_left_ipu, speech_left_talk])]

	elif region in ["region_2", "region_3", "region_4", "region_5"]:
		#set_of_behavioral_predictors = [speech_ipu, merge_dict ([speech_ipu, AUcs]),
										#speech_left_talk, speech_talk, merge_dict ([speech_left_ipu, speech_left_talk]),  merge_dict ([speech_left_talk, AUcs]),  merge_dict ([speech_left_ipu, speech_left_talk, AUcs]),
										#merge_dict ([speech_ipu, speech_talk, AUcs])]
		set_of_behavioral_predictors = [AUcs, speech_ipu, speech_left_ipu, speech_left_talk, speech_talk,
										merge_dict ([speech_left_ipu, AUcs]),  merge_dict ([speech_ipu, AUcs]),
										audio, merge_dict ([speech_ipu, audio]), audio_left, merge_dict ([speech_left_ipu, audio_left])]
		#set_of_behavioral_predictors = [speech_ipu, merge_dict ([speech_ipu]),
				#speech_left_talk, speech_talk, merge_dict ([speech_left_ipu, speech_left_talk]),  merge_dict ([speech_left_talk]),  merge_dict ([speech_ipu, speech_talk])]
		#set_of_behavioral_predictors = [AUcs]
	elif region in ["region_6", "region_7"]:
		set_of_behavioral_predictors = [speech_ipu, merge_dict ([speech_ipu, AUcs, face]),
										speech_talk, merge_dict ([speech_ipu, speech_talk]),  merge_dict ([speech_talk, AUcs, face]),  merge_dict ([speech_ipu, speech_talk, AUcs, face]),
										speech_items, all_1, gaze_angle, merge_dict ([speech_items, all_1, gaze_angle])]


	elif region in ["region_8", "region_9"]:
	    set_of_behavioral_predictors = [speech_items, all_1, gaze_angle, merge_dict ([speech_items, all_1, gaze_angle])]


	return set_of_behavioral_predictors
