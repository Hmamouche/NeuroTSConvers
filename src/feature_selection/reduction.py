import numpy as np
import pandas as pd

from src. feature_selection. fcbf import fcbf
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

def reduce_train_test (train_, test_, method, perc_comps = 1.0, n_comps = 0):

	"""
	Reduce the dimmensionality of the train and test data with model fitted on train data
	method : dimension reduction or feature selection methode used
	percs_comps : percentage of n_components in case of DR methods, and a threshold for the FCBF method.
	"""

	# make a copy to do no change the input data
	train = train_.copy ()
	test = test_. copy ()

	# compute number of features to select from percentage
	if n_comps == 0:
		n_comps = (train. shape[1] - 1) * perc_comps
		n_comps = int (n_comps)

	if method == "None":
	    return train, test, range (0, train. shape[1] - 1)

	elif method == "RFE":
	    return ref_local (train_, test_, n_comps)

	if method == "FCBF":
		sbest = fcbf (train [:, 1: ], train [:, 0], perc_comps)

		#best_indices = [int (a) for a in sbest[:, 1]]
		print (sbest)

		if len (sbest) == 0:
			return train, test, []

		sbest = [int (a) + 1 for a in sbest[:,1]]

		train = train [:,  [0] + sbest]
		test = test [:,  [0] + sbest]

		return train, test, sbest

	if method == "PCA":
	    model = PCA (n_components = n_comps, random_state = 5)

	elif method == "KPCA":
	    model = KernelPCA (n_components = n_comps)

	elif method == "IPCA":
	    model = IncrementalPCA (batch_size = None, n_components = n_comps)

	model = model.fit (train [:, 1: ])

	train = np.concatenate ((train [:, 0:1], model. transform (train [:, 1: ])), axis=1)
	test = np.concatenate ((test [:, 0:1], model. transform (test [:, 1: ])), axis=1)

	return train, test, [int (a) for a in range (train. shape[1] - 1)]

#===============================================================#
def ref_local (train, test, n_comp):

	# Prediction model to use
	estimator = RandomForestClassifier (n_estimators = 150, max_features = 'auto', bootstrap = True, max_depth = 10)
	selector = RFE (estimator, n_comp, step = 1)
	selector = selector.fit (train [:, 1: ], train [:, 0])

	support = selector. support_
	best_indices = []

	if len (support) == 0:
		best_indices = [i for i in range (train. shape [1] - 1)]
	else:
		for i in range (len (support)):
		    if support[i]:
		        best_indices. append (i)

	return (train [:, [0] + [(a + 1) for a in best_indices]], test [:, [0] + [(a + 1) for a in best_indices]], best_indices)

#=====================================================================
def rfe_reduction (train_, test_, percs):

	"""
	- Reduce train and test data with a model fitted on train data
		based on the recursive feature elimination method.
	- percs: percentage of features to select
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

	#return (train [:, [0] + best_indices], test [:, [0] + best_indices], best_indices, score)
	return (best_indices, score)


#======================================================================================
def manual_selection (region):

    # TODO: store information in a file instead of computing each time dictionaries
	AUrs = {"facial_features_ts": [" AU01_r"," AU02_r"," AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r"," AU23_r"," AU25_r"," AU26_r"]}
	AUcs = {"facial_features_ts": [" AU01_c"," AU02_c"," AU04_c"," AU05_c"," AU06_c"," AU07_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c"," AU20_c"," AU23_c"," AU25_c"," AU26_c"]}
	eyetracking =  {"eyetracking_ts": ["Vx", "Vy", "saccades", "Face", "Mouth", "Eyes"]}
	face =  {"eyetracking_ts": ["Face"]}

	head_pose_r = {"facial_features_ts": [" pose_Rx", " pose_Ry", " pose_Rz"]}
	head_pose_t = {"facial_features_ts": [" pose_Tx", " pose_Ty", " pose_Tz"]}
	gaze_angle = {"facial_features_ts": [" gaze_angle_x", " gaze_angle_y"]}
	colors = {"colors_ts": ["colorfulness"]}

	audio = {"speech_ts": ["Signal"]}
	audio_left = {"speech_left_ts": ["Signal_left"]}
	speech_activity = {"speech_ts": ["SpeechActivity"]}
	speech_activity_left = {"speech_left_ts": ["SpeechActivity_left"]}
	speech_ipu = {"speech_ts": ["IPU"]}
	speech_left_ipu = {"speech_left_ts": ["IPU_left"]}
	speech_left_disc_ipu = {"speech_left_ts": ["disc_IPU_left"]}
	speech_disc_ipu = {"speech_ts": ["disc_IPU"]}
	speech_talk = {"speech_ts": ["talk"]}
	speech_left_talk = {"speech_left_ts": ["talk_left"]}

	all_0 = AUrs
	all_1 = merge_dict ([AUrs, eyetracking, head_pose_r])
	all_2 = merge_dict ([eyetracking])
	all_3 = merge_dict ([face])
	all_4 = merge_dict ([eyetracking, head_pose_r, head_pose_t])
	all_5 = merge_dict ([face, head_pose_r, head_pose_t])
	all_6 = merge_dict ([head_pose_r, head_pose_t])
	all_7 = merge_dict ([AUrs, head_pose_r, head_pose_t])
	all_8 = merge_dict ([eyetracking, AUrs])
	all_9 = merge_dict ([face, head_pose_r, head_pose_t])


	items = {"speech_ts": ["FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters"]}
	emotions = {"speech_ts":["Polarity", "Subjectivity"]}
	lexicalR = {"speech_ts":["LexicalRichness1", "LexicalRichness2"]}
	items_left = {"speech_left_ts": ["FilledBreaks_left", "Feedbacks_left", "Discourses_left", "Particles_left", "Laughters_left"]}
	speech_items = merge_dict ([speech_ipu, items])
	speech_items_left = merge_dict ([speech_left_ipu, items_left])

	if region in ["Fusiform Gyrus", "LeftFusiformGyrus"]:
		set_of_behavioral_predictors = [all_1, all_8]
		#set_of_behavioral_predictors = [face, head_pose_r, all_0, all_1, all_2, all_3, all_4, all_5, all_6, all_7, all_8, all_9]

	elif region in ["LeftFrontaleyeField"]:
		#set_of_behavioral_predictors = [all_6]
		set_of_behavioral_predictors = [{"eyetracking_ts": ["x", "y"]}, eyetracking, {"eyetracking_ts": ["Vx", "Vy"]}, {"eyetracking_ts": ["saccades"]}]

	elif region in ["LeftVentralMotor", "LeftDorsalMotor"]:
		set_of_behavioral_predictors = [audio, audio_left, speech_left_ipu, speech_ipu, speech_talk, speech_left_talk,
										merge_dict ([speech_ipu, speech_left_ipu]),
										merge_dict ([audio_left, speech_talk]),
										merge_dict ([speech_left_ipu, audio_left])]

	elif region in ["Left Motor Cortex", "Right Motor Cortex"]:
		#set_of_behavioral_predictors = [audio, audio_left, speech_left_ipu, speech_ipu, speech_talk, speech_left_talk, merge_dict ([speech_ipu, speech_left_ipu]),
										#merge_dict ([speech_ipu, speech_talk]),  merge_dict ([speech_left_ipu, speech_left_talk]), merge_dict ([speech_left_ipu, AUcs]), merge_dict ([speech_left_ipu, AUrs])]
		#set_of_behavioral_predictors = [speech_left_ipu]

		set_of_behavioral_predictors = [audio, speech_activity_left, audio_left, speech_left_ipu, speech_ipu, speech_talk, speech_left_talk, speech_left_disc_ipu,
		merge_dict ([speech_left_disc_ipu, speech_disc_ipu]),
		merge_dict ([speech_ipu, speech_left_ipu]),
		merge_dict ([speech_activity_left, speech_left_talk]),
		merge_dict ([speech_left_ipu, speech_left_talk]),
		merge_dict ([speech_left_ipu, {"speech_left_ts":["Overlap_left"]}]),
		merge_dict ([speech_left_ipu, {"speech_left_ts":["Overlap_left", "ReactionTime_left"]}]),
		merge_dict ([speech_left_disc_ipu, {"speech_left_ts":["Overlap_left", "ReactionTime_left"]}])
		]

	elif region in ["Left Superior Temporal Sulcus", "Right Superior Temporal Sulcus"]:

		set_of_behavioral_predictors = [speech_items, speech_disc_ipu, speech_ipu, speech_left_ipu,
										merge_dict([speech_items, AUrs]),
										merge_dict ([speech_ipu, AUrs]), audio,
										merge_dict ([speech_ipu, audio]),
										merge_dict ([speech_left_ipu, audio_left]),
										merge_dict ([speech_disc_ipu, items, emotions, lexicalR]),
										merge_dict ([speech_ipu, items, emotions])]

	elif region in ["region_6", "region_7"]:
		set_of_behavioral_predictors = [speech_items, emotions, eyetracking,
										merge_dict ([speech_items, emotions]),
										merge_dict ([speech_items, eyetracking]),
										merge_dict ([emotions, eyetracking]),
										merge_dict ([speech_items, emotions, eyetracking]),
										merge_dict ([speech_items, emotions, eyetracking])]


	elif region in ["region_8", "region_9"]:
	    set_of_behavioral_predictors = [speech_items, all_1, merge_dict ([speech_items, all_1, gaze_angle])]


	return set_of_behavioral_predictors
