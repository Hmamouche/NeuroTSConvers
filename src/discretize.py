import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

import argparse

#=====================================================

def normalize (M):
	minMax = np.empty ([M.shape[1], 2])
	for i in range(M.shape[1]):
		#print (M[:,i])
		max = np.max(M[:,i])
		min = np.min(M[:,i])
		minMax[i,0] = min
		minMax[i,1] = max

		if min < max:
			for j in range(M.shape[0]):
				M[j,i] = (M[j,i] - min) / (max - min)

	return minMax

#=====================================================

def find_peaks_ (y, height = 0):
	x = []
	for i in range (len (y)):
		if y[i] > height:
			x.append (i)
	return x

#===============================================================
def discretize_array (df, min = 0.1, mean = False, peak = False):

	cols = df. columns
	M = df. values

	for j in range (1, M.shape [1]):

		if peak:
			peaks, _ = find_peaks (M[:,j], height=0)
			for i in range (M.shape [0]):
				M[i,j] = 0

			for x in peaks:
				M[x,j] = 1

		else:
			for i in range (M.shape [0]):
				if mean:
					min = np. mean (M[:,j])
				if M[i,j] <= min:
					M[i, j] = 0.0
				else:
					M[i, j] = 1.0

	return pd.DataFrame (M, columns = cols)


#==============================================================
def discretize_df (df, min = 0.1, n_classes = 2, mean = False):

	if mean:
		for i in range (1, df. shape [1]):
			x = df. iloc [:,i] .values
			peaks = find_peaks_ (x, height = np.mean (x))
			df. iloc [:, i] = 0.0
			df. iloc [peaks, i] = 1.0

	elif n_classes == 3:
		df [df <= 0] = 0
		df [df > min] = 2
		df [((df <= min) & (df > 0))] = 1

	elif n_classes == 2:
		df [df <= float (min)] = 0.0
		df [df > float (min)] = 1.0

	elif n_classes == "peak":
		for i in range (1, df. shape [1]):
			x = df. iloc [:,i] .values
			peaks = find_peaks_ (x, height = min)
			df. iloc [:, i] = 0.0
			df. iloc [peaks, i] = 1.0

	return df

#=====================================================

def discretize_df_kmeans (df, k = 3):
	for col in df. columns [1:]:
		clustering = KMeans (n_clusters = k, random_state = 1). fit (df. loc[:, col]. values. reshape (-1, 1))
		'''print (clustering. inertia_)
		exit (1)'''
		#clustering = DBSCAN (eps=3, min_samples=2).fit (df. loc[:, col]. values. reshape (-1, 1))
		df [col] = clustering. labels_

#=====================================================
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("data_type", help="data type")
	parser.add_argument("--nbins", "-k", default = 2, type = int)
	parser.add_argument("--mean", "-mean",  action="store_true")
	parser.add_argument("--peak", "-peak",  action="store_true")
	parser.add_argument("--threshold", "-min", default = 0.0, type = float)
	parser.add_argument("--type", "-t", default = "raw")
	args = parser.parse_args()

	""" store the discretization parameters """
	f= open("disc_params.txt","w+")
	for item in vars (args). items ():
		f. write ("%s: %s\n"%(item[0], item [1]))
	f. close ()


	subjects_in = glob ("time_series/*")
	subjects_out = glob ("time_series/*")

	#=====================================================#
	""" discretize physiological data """
	if args.data_type == "p":

		if args. type == "raw":
			in_data_type = "/new_physio_ts"
		elif args. type == "diff":
			in_data_type = "/physio_diff_ts"
		elif args. type == "smooth":
			in_data_type = "/physio_smooth_ts"

		for i in range (len (subjects_in)):

			if not os.path. exists ("%s/discretized_physio_ts"%subjects_in[i]):
				os.makedirs ("%s/discretized_physio_ts"%subjects_in[i])

			subjects_in[i] = subjects_in[i] + in_data_type
			subjects_out[i] = subjects_out[i] + "/discretized_physio_ts"

			pkl_files = glob ("%s/*pkl"%subjects_in[i])
			pkl_files. sort ()

			for filepath in pkl_files:
				df = pd. read_pickle (filepath)
				filename = filepath. split('/')[-1]

				if args.threshold == -1.0:
					discretize_df_kmeans (df, k = args. nbins)

				else:
					#discretize_df (df, float (args.threshold), n_classes = args. nbins)
					df = discretize_array (df, float (args.threshold), args. mean, args. peak)
				df.to_pickle ("%s/%s" %(subjects_out[i], filename))

	#=======================================================#
	""" discretize colors data """
	if args.data_type == "c":
		for i in range (len (subjects_in)):

			if not os.path. exists ("%s/discretized_colors_ts"%subjects_in[i]):
				os.makedirs ("%s/discretized_colors_ts"%subjects_in[i])

			subjects_in[i] = subjects_in[i] + "/colors_ts"
			subjects_out[i] = subjects_out[i] + "/discretized_colors_ts"

			pkl_files = glob ("%s/*pkl"%subjects_in[i])

			for filepath in pkl_files:
				df = pd. read_pickle (filepath)
				filename = filepath. split('/')[-1]
				discretize_df_kmeans (df, k = 7)
				df.to_pickle ("%s/%s" %(subjects_out[i], filename))

	#=====================================================
	# discretize transcription data
	elif args.data_type == "t":
		for i in range (len (subjects_in)):

			if not os.path. exists ("%s/discretized_speech_ts"%subjects_in[i]):
				os.makedirs ("%s/discretized_speech_ts"%subjects_in[i])

			subjects_in[i] = subjects_in[i] + "/speech_ts"
			subjects_out[i] = subjects_out[i] + "/discretized_speech_ts"

			pkl_files = glob ("%s/*pkl"%subjects_in[i])

			for filepath in pkl_files:
				df = pd. read_pickle (filepath)
				colnames = df.columns
				df = df. values
				normalize (df)

				df = np. digitize (df, [0, 0.01, 0.2, 0.4, 0.6, 0.8, 1])
				df = pd.DataFrame (df, columns = colnames)

				#print (df)
				#exit (1)

				filename = filepath. split('/')[-1]
				df.to_pickle ("%s/%s" %(subjects_out[i], filename))
