# coding: utf8

import numpy as np
import pandas as pd
import seaborn

import sys
import os

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def usage():
	print ("execute the script with -h for usage.")

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("video", help="the path of the video to process.")
	parser.add_argument("out_dir", help="the path where to store the results.")
	parser.add_argument("--show",'-s', help="Showing the video.", action="store_true")
	parser.add_argument("--openface",'-op', help="Showing the video.", default="../../OpenFace")

	args = parser.parse_args()

	if args. out_dir == 'None':
		usage ()
		exit ()

	if args. out_dir[-1] != '/':
		args. out_dir += '/'

	if args. openface [-1] != '/':
		args. openface += '/'

	# Input directory
	conversation_name = args. video.split ('/')[-1]. split ('.')[0]
	out_file = args. out_dir + conversation_name

	# verify if the file already exists
	if os.path.isfile ("%s.pkl"%out_file) and os.path.exists ("%s"%out_file):
		test_df = pd.read_pickle ("%s.pkl"%out_file)
		csv_df = pd.read_csv ("%s/%s.csv"%(out_file, conversation_name), sep = ',', header = 0)
		if test_df. shape [0] == 50 and csv_df. shape [0] == 1799:
			print ("Already processed")
			exit (1)
	# Run OpenFace binary program to the video and the given output directory
	os. system (args. openface + "build/bin/FeatureExtraction -q -f %s -out_dir %s" %(args. video, out_file))

	# read csv file
	openface_data = pd. read_csv ("%s/%s.csv"%(out_file, conversation_name), sep=',', header=0)

	# Keeping just  some features : gaze, head pose, and facial unit actions
	movements_cols = [" timestamp", " gaze_angle_x", " gaze_angle_y", " pose_Tx", " pose_Ty", " pose_Tz", " pose_Rx", " pose_Ry", " pose_Rz"]
	action_units_cols = [" AU01_r"," AU02_r"," AU04_r"," AU05_r"," AU06_r"," AU07_r"," AU09_r"," AU10_r"," AU12_r"," AU14_r"," AU15_r"," AU17_r"," AU20_r"," AU23_r", " AU25_r", " AU26_r", " AU45_r"]
	action_units_existence_cols = [" AU01_c"," AU02_c"," AU04_c"," AU05_c"," AU06_c"," AU07_c"," AU09_c"," AU10_c"," AU12_c"," AU14_c"," AU15_c"," AU17_c"," AU20_c"," AU23_c", " AU25_c", " AU26_c", " AU45_c"]
	land_marks_cols = [" x_%d"%i for i in range (68)] + [" y_%d"%i for i in range (68)]

	head_movement = openface_data .loc [:, movements_cols]
	land_marks = openface_data .loc [:, land_marks_cols]
	action_units = openface_data .loc [:, action_units_cols]
	actions_unis_existence = openface_data. loc [:, action_units_existence_cols]

	#action_units = action_units. multiply (actions_unis_existence)

	data = pd.concat ([head_movement, actions_unis_existence, action_units, land_marks], axis=1). values

	# eliminate space from colnames
	cols = ["Time (s)"] + movements_cols + action_units_existence_cols + action_units_cols + land_marks_cols

	aggregated_time_series = []

	begin_time = data [0,0]
	begin_index = 0
	discretized_time = 0
	rows_ts = []

	# Construct the index: 50 observations in physiological data
	index = [0.6025]
	for i in range (1, 50):
		index. append (1.205 + index [i - 1])

	j = 0
	for i in range (data. shape [0]):
		if j >= 50:
			break
		if (data[i,0] > index [j]):
			aggregated_time_series. append ([index [j]] + np.nanmax (rows_ts[1:], axis=0). tolist ())
			discretized_time += 1.205
			initializer = 0
			begin_time = data[i,0]
			begin_index = i
			rows_ts = []
			j+= 1

		rows_ts. append (data [i,:])

	if len (rows_ts) > 0 and j < 50:
		aggregated_time_series. append ([index [j]]+ np.amax (rows_ts[1:], axis=0). tolist ())

	aggregated_time_series = pd.DataFrame (aggregated_time_series)
	aggregated_time_series. columns = cols
	aggregated_time_series. to_pickle ("%s.pkl"%out_file)


	os. system (" rm -r %s/*aligned" %(out_file))
	os. system (" rm %s/*.avi* " %(out_file))
	os. system (" rm %s/*.hog* " %(out_file))
