"""
	Author: Youssef Hmamouche
	Year: 2019
	Compute features about head movment energy, contractions (AUs), head movements ...
"""

import sys, os, inspect, argparse
import numpy as np
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
maindir = os.path.dirname(parentdir)

sys.path.insert (0, maindir)
import src.resampling as resampling

#===================================================
def moving_average(n_signal, periods=3):

	if n_signal. shape[1] == 0 or n_signal. shape [0] == 0:
		raise ("Error in moving_average, input signal is emtpy or not an n-dimmensional array.")

	weights = np.ones(periods) / periods
	res = np.convolve(n_signal[:,0], weights, mode='valid'). reshape ((-1,1))

	for j in range (1, n_signal. shape [1]):
		res = np. insert (res, res. shape [1], np.convolve(n_signal[:,j], weights, mode='valid'), axis = 1)

	return res

#================================================#
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("video", help = "the path of the video to process.")
	parser.add_argument("out_dir", help = "the path where to store the results.")
	parser.add_argument("--demo",'-d', help = "If this script is using for demo.", action = "store_true")
	parser.add_argument("--facial_features",'-faf', help = "the path of the facial features file (csv file) (for demo).")
	args = parser.parse_args()

	if args. out_dir[-1] != '/':
		args. out_dir += '/'

	# This for an independant utilization of the script (for demonstration)
	if args. demo:
		openface_file = args. facial_features
		conversation_name = "facial_features_energy"
	else:
		# subject number from video path
		subject = args. video.split ('/')[-2]
		conversation_name = args. video.split ('/')[-1]. split ('.')[0]
		openface_file = "time_series/%s/openface_ts/%s.pkl"%(subject, conversation_name)

	# Input directory and output file
	out_file = args. out_dir + conversation_name
	print (out_file)

	# check if file already processed
	if os.path.exists (out_file + '.pkl'):
		print (out_file)
		print ("Warning, file already processed")
		exit (1)

	# check if feature extraction (openface) has been processed
	if not os.path.exists (openface_file):
		print ("Error, file %s does not exists"%openface_file)
		exit (1)

	# The index of BOLD signal
	physio_index = [0.6025]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	# read  openface file
	openface_data = pd. read_pickle (openface_file)
	openface_data = openface_data[openface_data[" confidence"] > 0.5]

	#print (openface_data. loc [:," timestamp"])
	# feature extraction
	video_index = openface_data. loc [:," timestamp"]. values

	# First part of the variables
	df1 = pd.DataFrame ()
	df1 ["Time (s)"] = video_index
	df1 ["AUs_mouth_I"] = openface_data. loc [:,[" AU10_r", " AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r", " AU23_r", " AU25_r", " AU26_r"]]. sum (axis = 1)
	df1["AU_eyes_I"] = openface_data. loc [:, [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r", " AU09_r"]]. sum (axis = 1)
	df1["AU_all_I"] = df1 ["AUs_mouth_I"] + df1 ["AU_eyes_I"]

	# resampling
	output_time_series = pd.DataFrame (resampling. resample_ts (df1. values, physio_index, mode = "mean"), columns = df1.columns)

	# direct gaze
	direct_gaze_brut = openface_data. loc [:,[" timestamp"," gaze_angle_x", " gaze_angle_y"]]. values
	direct_gaze = moving_average (direct_gaze_brut, 30)
	# Re-add the first 29 observations lost by moving average
	for i in range (29):
		direct_gaze = np. insert (direct_gaze, i, direct_gaze_brut[i], axis = 0)

	direct_gaze[:,1] = direct_gaze[:,1] - np. mean (direct_gaze[:,1])
	direct_gaze[:,2] = direct_gaze[:,2] - np. mean (direct_gaze[:,2])

	for i in range (len (direct_gaze)):
		direct_gaze[i,1] = np. sqrt (direct_gaze[i,1]**2 + direct_gaze[i,2]**2)
		if direct_gaze[i,1] < 0.05:
			direct_gaze[i,1] = 1
		else:
			direct_gaze[i,1] = 0

	direct_gaze = np. delete (direct_gaze, 2, 1)
	direct_gaze = resampling. resample_ts (direct_gaze, physio_index, mode = "mean")
	output_time_series["Direct_gaze_I"] = direct_gaze[:,1]

	# head positions
	head_positions = openface_data. loc [:,[" timestamp", " pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz"]]

	# very specific to our corpus: the robot is fix, we force detected variables to be constant
	conv_type = conversation_name. split ('_')[-2]
	if conv_type == "CONV2":
		for j in range (1, 7):
			head_positions. iloc [:,j] = 0
		output_time_series. loc [:, "Direct_gaze_I"] = 1

	else:
		# stabilize openface errors
		for j in range (1, 7):
			min_ = head_positions. iloc [3:,j]. min ()
			max_ = head_positions. iloc [3:,j]. max ()

			if j < 4:
				# translation (mm)
				seuil = 20
			else:
				# rotation in radian
				seuil = 0.1

			if ((max_ - min_) <= seuil):
				head_positions. iloc [:,j] = 0

	#resampling
	#print (pd.DataFrame (head_positions))
	head_positions_diff = resampling. resample_ts (head_positions.values, physio_index, mode = "mean")
	# computing the differences
	head_positions_diff = np. gradient (head_positions_diff[:,1:], head_positions_diff[:,0], axis = 0)

	output_time_series["Head_Tx_I"] = head_positions_diff [:,0]
	output_time_series["Head_Ty_I"] = head_positions_diff [:,1]
	output_time_series["Head_Tz_I"] = head_positions_diff [:,2]

	output_time_series["Head_Rx_I"] = head_positions_diff [:,3]
	output_time_series["Head_Ry_I"] = head_positions_diff [:,4]
	output_time_series["Head_Rz_I"] = head_positions_diff [:,5]

	# head movment gradient
	head_translation = np. gradient (head_positions. loc [:, [" pose_Tx", " pose_Ty", " pose_Tz"]]. values[1:,:] * 0.001, video_index[1:], axis = 0) #0.001: mm to m
	head_rotation = np. gradient (head_positions. loc [:, [" pose_Rx", " pose_Ry", " pose_Rz"]]. values[1:,:], video_index[1:], axis = 0)

	# head movment square of gradient and sum over axex x,y, z
	head_translation = np. sum (np. square (head_translation), axis = 1). reshape ((-1,1))
	head_rotation =  np. sum (np. square (head_rotation), axis = 1). reshape ((-1,1))

	# add timestamp as first column
	head_translation = np. insert (head_translation, 0, video_index[1:], axis = 1)
	head_rotation = np. insert (head_rotation, 0, video_index[1:], axis = 1)

	# resample by the sum the square differences (representing movement quantity)
	head_translation_energy = resampling. resample_ts (head_translation, physio_index, mode = "mean",  rotation = False, pixel = False)
	head_rotation_energy = resampling. resample_ts (head_rotation, physio_index,  mode = "mean", rotation = True, pixel = False)

	# concatenate all columns
	output_time_series["Head_R_movement_quantity_I"] = head_rotation_energy[:,1] #without index
	output_time_series["Head_T_movement_quantity_I"] = head_translation_energy[:,1]

	#print (output_time_series["Head_T_movement_quantity_I"])
	#exit (1)

	# save data in pickle file
	output_time_series.to_pickle (out_file + ".pkl")
