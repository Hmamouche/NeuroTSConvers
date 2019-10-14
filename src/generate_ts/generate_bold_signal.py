#import scipy.io as sio
import pandas as pd
import numpy as np
from glob import glob
import os
import sys
import argparse
from mat4py import loadmat
from skimage import filters
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import preprocessing
#-----------------------------------------------------------------------------
# Plot dataframe
def plot_df (df, labels, figsize=(12,9), y_lim = [0,1.2]):
	fig = plt.figure(figsize = figsize)

	plt.title('Title!', color='black')
	df. plot (y=labels, sharex=True, subplots=True, xticks = df.index, fontsize = 7, grid=False, legend= False, ax=fig.gca())

	ld = fig.legend(labels = labels,
	       loc='upper right',   # Position of legend
	       prop={'size':6},
	       ncol=1,
	       fontsize = 'small')

	fig.text (0.5, 0.12, 'Time (s)', ha='center')
	fig.text (0.07, 0.5, 'Series', va='center', rotation='vertical')
	plt. show ()


#============================================================
def normalize_vect (x):
	max = np.max (x)
	min = np.min (x)

	if min < max:
		for i in range (len (x)):
			x[i]= (x[i] - min) / (max - min)

#============================================================
def get_sliding_windows (n, size = 10):
	winds = []
	for j in range (n + 1 - size):
		winds. append (range (j, j + size))

	return winds

#===========================================================
def discretize_vect (x, threshold):
	result = []
	for i in range (len (x)):
		if x[i] > threshold:
			result. append ([1])
		else:
			result. append ([0])

	return result
#============================================================
def discretize_vect_sliding (x, win_size = 10):
	mat = []
	winds = get_sliding_windows (len (x), 10)

	for i in range (len (x)):
		values_in_windows = []
		for j in range (len (winds)):
			if i in winds[j]:
				indice_i = winds[j]. index (i)
				row = [x[k] for k in winds[j]]
				#normalize_vect (row)

				if row [indice_i] > 0.5:
					values_in_windows. append (1)
				else:
					values_in_windows. append (0)

		# count the number of 1 in each moving window
		if (float (np. sum (values_in_windows)) / len (values_in_windows) > 0.5):
			mat. append (1.0)
		else:
			mat. append (0.0)
	return mat

#======================================================

def nearestPoint(vect, value):
    dist = abs(value - vect[0])
    pos = 0

    for i in range(1, len(vect)):
        if abs(value - vect[i]) < dist:
            dist = abs(value - vect[i])
            pos = i

    return pos

#======================================================

def add_duration(df):
    duration = []

    for i in range(df.shape[0]):
        if i == (df.shape[0] - 1):
            duration.append([df.iloc[i, 3] / 1000.0, df.iloc[i, 3] / 1000.0, 0])
        else:
            duration.append([df.iloc[i, 3] / 1000.0, df.iloc[i + 1, 3] / 1000.0,
                             df.iloc[i + 1, 3] / 1000.0 - df.iloc[i, 3] / 1000.0])

    df = pd.concat([df, pd.DataFrame(duration, columns=['begin', 'fin', 'Interval'])], axis=1)

    return df

#======================================================

def convers_to_df (data, data_discretized, colnames, index, begin, end, type_conv, num_conv):

	index_normalized = index[begin:end]
	start_pt = index_normalized [0]

	for j in range(0, end - begin):
	   index_normalized[j] -= start_pt

	convers_data = pd.DataFrame ()
	convers_data ["Time (s)"] = index_normalized

	convers_data_discr = pd.DataFrame ()
	convers_data_discr ["Time (s)"] = index_normalized

	for i in range (len (colnames)):
		convers_data. loc [:, colnames [i]] = data [begin : end, i]
		convers_data_discr. loc [:, colnames [i]] = data_discretized [begin : end, i]

	filename = "time_series/" + subject + "/physio_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"
	filename_discr = "time_series/" + subject + "/discretized_physio_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"

	convers_data.to_pickle (filename)
	convers_data_discr.to_pickle (filename_discr)

#======================================================

if __name__ == '__main__':

	mat_data = loadmat("data/ROI_PhysSocial_Youssef.mat")
	index = [0.6025]
	for i in range (1, 4 * 385):
	    index. append (1.205 + index [i - 1])


	""" store the discretization parameters """
	f= open("disc_params.txt","w+")
	f. write ("Thierry discretization")

	# loop over subjects
	for s in range (25):
		subject = "sub-%02d"%(s + 1)
		print (subject)

		for fname in ["physio_ts", "discretized_physio_ts"]:
			if not os.path.exists ("time_series/%s/%s"%(subject, fname)):
			    os.makedirs("time_series/%s/%s"%(subject, fname))

		#----------------------------------------------------#
		#------------  Concatenate all objects   ------------#
		#----------------------------------------------------#
		bol_signal = pd. DataFrame ()
		bold_signal_discretized = pd. DataFrame ()
		# loop over regions
		for r in range (4):
			region_name = mat_data ["results"][s]["roi"][r]['name']
			sessions =  mat_data ["results"][s]["roi"][r]['sess']

			#disc_threshold = min ([sessions[l][0] for l in range (4)])
			#f. write ("%s: %d"%(region_name, disc_threshold))

			# loop over sessions
			for i in range (4):

				#disc_threshold = filters. threshold_otsu ( np. array (sessions [i]))
				#disc_threshold = np. mean (np. array (sessions [i]))
				norm = sessions [i][:]
				#normalize_vect (norm)
				#disc_threshold = filters. threshold_otsu (np. array (norm))

				# discretize the bold signal of each session
				#clustering = KMeans (n_clusters = 2, random_state = 1). fit (np.array (norm). reshape (-1, 1))
				# norm = clustering. labels_

				normalize_vect (norm)
				discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')
				norm_disc = discretizer.fit_transform(norm). flatten ()

				#disc_threshold = norm[0]
				#norm_disc = discretize_vect (norm, np. mean (norm))

				if i == 0:
					region_data = np. array (norm)
					region_data_discretized = np. array (norm_disc)

				else:
					region_data = np. concatenate ((region_data, np. array (norm)), axis = 0)
					region_data_discretized = np. concatenate ((region_data_discretized, norm_disc), axis = 0)


			bol_signal [region_name] = region_data. flatten ()
			bold_signal_discretized [region_name] = region_data_discretized. flatten ()

		#plot_df (bol_signal, bold_signal_discretized. columns, figsize=(12,9))
		#exit (1)

		if bol_signal. shape[0] == 0:
			continue

		colnames = bol_signal. columns
		bol_signal = bol_signal. values
		bold_signal_discretized = bold_signal_discretized. values

		testBlocks = ["TestBlocks" + str (i + 1) for i in range (4)]


		#--------------------------------------------------------#
		indice_block = 0
		for testBlock in testBlocks:
			logfile = glob ("data/physio_data/logfiles/*" + subject + "_task-convers-" + testBlock + "*")
			if len (logfile) == 0:
				continue
			else:
				logfile = logfile [0]

			df = pd.read_csv (logfile, sep='\t', header=0)
			df = df [['condition', 'image', 'duration', 'ONSETS_MS']]

			df = add_duration(df)

			hh_convers = df [df.condition.str.contains("CONV1")] [['condition', 'begin', 'fin']]
			hr_convers = df [df.condition.str.contains("CONV2")] [['condition', 'begin', 'fin']]

			nb_hh_convers = hh_convers. shape [0]
			nb_hr_convers = hr_convers. shape [0]

			hh = 1
			hr = 2

			for i in range(nb_hh_convers):
			    begin = nearestPoint (index, hh_convers.values[i][1]) + (385 * indice_block)
			    end = nearestPoint (index, hh_convers.values[i][2]) + (385 * indice_block)
			    convers_to_df (bol_signal, bold_signal_discretized, colnames, index, begin, end, "CONV1", hh)
			    hh += 2


			for i in range(nb_hr_convers):
			    begin = nearestPoint (index, hr_convers.values[i][1]) + (385 * indice_block)
			    end = nearestPoint (index, hr_convers.values[i][2]) + (385 * indice_block)
			    convers_to_df (bol_signal, bold_signal_discretized, colnames, index, begin, end, "CONV2", hr)
			    hr += 2
			indice_block += 1
	f. close ()
