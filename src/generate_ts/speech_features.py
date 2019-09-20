# coding: utf8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.io.wavfile as wav
from scipy.signal import hilbert, chirp  # for the envelope of the signal
import spacy as sp

import utils.tools as ts
#from ..tools import plot_df

import sys
import glob
import os
import argparse

sys.path.insert(0,'src/utils/SPPAS')

import utils.SPPAS.sppas.src.anndata.aio.readwrite as spp


#-----------------------------------------------------------------------------
OK_FORMS = [u"o.k.",u"okay",u"ok",u"OK",u"O.K."]
VOILA_FORMS = [u"voilà",u"voila"]
DACCORD_FORMS = [u"d'accord",u"d' accord"]
LAUGHTER_FORMS = [u'@',u'@ @',u'@@']
EMO_FORMS = [u'@',u'@ @',u'@@',u'ah',u'oh']

REGULATORY_DM_SET = set([u"mh",u"ouais",u"oui",u"o.k.",u"okay",u"ok",u"OK",u"O.K.",u"d'accord",u"voilà",u"voila",u'bon',u"d'",
u"accord",u'@',u'@ @',u'@@',u'non',u"ah",u"euh",u'ben',u"et",u"mais",u"*",u"heu",u"hum",u"donc",u"+",u"eh",u"beh",u"donc",u"oh",u"pff",u"hein"])

FILLED_PAUSE_ITEMS = [u"euh",u"heu",u"hum",u"mh"]
SILENCE = [u'+',u'#',u'',u'*']
LAUGHTER = [u'@',u'@@']

MAIN_FEEDBACK_ITEMS = [u"mh",u"ouais",u"oui",u'non',u'ah',u"mouais"]+ OK_FORMS + VOILA_FORMS + DACCORD_FORMS + LAUGHTER_FORMS
MAIN_DISCOURSE_ITEMS = [u"alors",u"mais",u"donc",u'et',u'puis',u'enfin',u'parceque',u'parcequ',u'ensuite']
MAIN_PARTICLES_ITEMS = [u"quoi",u"hein",u"bon",u'mais',u'ben',u'beh',u'enfin',u'vois',u'putain',u'bref']

colors = ["black", "darkblue", "brown", "red", "slategrey", "darkorange", "grey","blue", "indigo", "darkgreen"]

#-----------------------------------------------------------------------------
def usage():
	print ("execute the script with -h for usage.")

#-----------------------------------------------------------------------------

def get_intervals (ts):
	y = []
	for i in range (0, len (ts[0]) - 1):
		if (ts[1][i] == 1 and ts[1][i+1] == 1):
			y.append ( [ts[0][i], ts[0][i + 1]] )
	return y

#---------------------------------------------------------------------
# Get nearest index point in vect to the the value 'value'
def nearestPoint (vect, value):
	index = -1
	for i in range (len (vect) - 1):
		if value >= vect [i] and value < vect [i + 1]:
			index = i
			break
	if value == vect[-1]:
		index = len (vect) - 1
	return index

#---------------------------------------
# Get points from the time series ts that correspond to index axis
# two modes are available: the sum or the mean of points in previous interval
def sample_cont_ts (ts, axis, mode = 'sum'):
	set_of_points = [[] for x in range (len (axis))]
	y = [0 for x in range (len (axis))]

	for i in range (len (ts[0])):
		for j in range (0, len (axis)):
			if j == 0:
				if ((0 < ts[0][i]) and (axis [j] >= ts[0][i])):
					set_of_points[j]. append (ts[1][i])
					break
			else:
				if ((axis [j - 1] < ts[0][i]) and (axis [j] >= ts[0][i])):
					set_of_points[j]. append (ts[1][i])
					break

	if mode == 'mean':
		for j in range (0, len (y)):
			if len (set_of_points[j]) > 0:
				y[j] = np.mean (set_of_points[j])

	if mode == 'max':
		for j in range (0, len (y)):
			if len (set_of_points[j]) > 0:
				y[j] = np.max (set_of_points[j])

	if mode == 'sum':
		for j in range (0, len (y)):
			if len (set_of_points[j]) > 0:
				y[j] = np.sum (set_of_points[j])

	return y

#---------- intersection between two intervals
def get_intersection (A, B):

	overlap = [0, 0]
	if A[1] <= B[0] or B[1] <= A[0]:
		return []
	else:
		overlap = [max (A[0], B[0]), min (A[1], B[1])]
	return overlap
#-----------------------------------------------------------------
# quantize time series
# step : the step between two observarions
# nb_obs : the number of observations
# axis : discrete vector
def sample_square_ts (ts, axis):

	axis_intervals = []
	for i in range (len (axis)):
		if i == 0:
			axis_intervals. append ([i, [0, axis[i]]])
		else:
			axis_intervals. append ([i, [axis[i-1], axis[i]]])



	y = [0 for i in range (len (axis))]


	# Transform the time series into a set of intervals
	intervals = get_intervals (ts)

	# We have tow cases: the interval is larger or smaller than the discretized step (1.205s)
	if len (intervals) == 0:
		return y

	for inter_ax in axis_intervals:
		step = inter_ax[1][1] - inter_ax[1][0]
		for interval in intervals:
			overlap = get_intersection (inter_ax [1], interval)

			if len (overlap) > 0:
				y [inter_ax [0]] += (overlap [1] - overlap [0]) / step

	return y

#========================================================
def new_sample_square (ts, axis):
	axis_intervals = []
	for i in range (len (axis)):
		if i == 0:
			axis_intervals. append ([0, axis[i] ])
		else:
			axis_intervals. append ([axis[i-1], axis[i] ])

	y = [0 for i in range (len (axis))]

	if len (ts) == 0:
		return y

	#print (axis_intervals)
	# Compute the durations of events in each interval of the axis
	i = 0
	for inter_ax in axis_intervals:
		step = inter_ax [1] - inter_ax [0]
		for interval in ts:
			overlap = get_intersection (inter_ax, interval)

			if len (overlap) > 0:
				y [i] += (overlap [1] - overlap [0]) # / step
		i += 1

	return y

#-------------------------------------------------------#
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("data_dir", help="the path of the file to process.")
	parser.add_argument("out_dir", help="the path where to store the results.")
	parser.add_argument("--left", "-l", help="Process participant speech.", action="store_true")

	args = parser.parse_args()

	data_dir = args. data_dir
	out_dir = args. out_dir

	if out_dir == 'None':
		usage ()
		exit ()

	if args. out_dir[-1] != '/':
		args. out_dir += '/'

	# create output dir file if does not exist
	if not os.path.exists (args. out_dir):
		os.makedirs (args. out_dir)

	filename = args. data_dir.split('/')[-1]. split ('.')[0]

	conversation_name = data_dir.split ('/')[-1]

	if conversation_name == "":
		if args. left:
			conversation_name = "speech_features_left"
		else:
			conversation_name = "speech_features"

	print ("-----------------", conversation_name)

	output_filename_1 = out_dir +  conversation_name + ".png"
	output_filename_pkl = out_dir +  conversation_name + ".pkl"

	#print ("Processing %s" %data_dir.split ('/')[-1])

	#if os.path.isfile (output_filename_1)  and os.path.isfile (output_filename_pkl):
	if os.path.isfile (output_filename_pkl):
		print ("Conversation already processed")
		exit (1)

	# Read audio, and transcription file
	for file in glob.glob(data_dir + "/*"):
		if "left-reduc.TextGrid" in file:
			transcription_left = file
		elif "right-filter.TextGrid" in file:
			transcription_right = file

		elif "right-filter-palign.textgrid" in file:
			transcription_right_palign = file

		elif "left-reduc-palign.textgrid" in file:
			transcription_left_palign = file

		elif ".wav" in file:
			if args.left:
				if "left-reduc.wav" in file:
					rate, signal = wav.read (file)
			else:
				if "right-filter.wav" in file:
					rate, signal = wav.read (file)

	analytic_signal = hilbert(signal)
	envelope = np. abs (analytic_signal). tolist ()
	step =  1.0 / rate

	signal_x = [0.0]
	for i in range (1, len (signal)):
		signal_x. append (step + signal_x[i-1])

	# Select language
	nlp = sp.load('fr_core_news_sm')

	# Read TextGrid files
	# get the left part of the transcription

	parser = spp.sppasRW (transcription_left)
	tier_left = parser.read(). find ("Transcription")
	if  tier_left is None:
		tier_left = parser.read(). find ("IPUs")

	# get the right part of the transcription
	parser = spp.sppasRW (transcription_right)
	tier_right = parser.read(). find ("Transcription")
	if  tier_right is None:
		tier_right = parser.read(). find ("IPUs")



	if args.left:
		tier = tier_left
	else:
		tier = tier_right

	# Silence
	# Index variable
	physio_index = [0.6025]
	#physio_index = [0]
	for i in range (1, 50):
		physio_index. append (1.205 + physio_index [i - 1])

	#silence = ts. get_silence (tier, physio_index, 1)
	#IPU = ts. get_ipu (tier, 1)
	#print (IPU)
	IPU, _ = ts. new_get_ipu (tier, 1)
	talk = physio_index, ts. get_dicretized_ipu (tier, physio_index, 1)

	# Overlap
	overlap = ts. get_overlap_new (tier_left, tier_right)

	# recation time
	reaction_time = ts. get_reaction_time (tier_left, tier_right)
	# Lexical richness
	richess_lex1 = ts.generate_RL_ts (tier, nlp, "meth1")
	richess_lex2 = ts.generate_RL_ts (tier, nlp, "meth2")

	# Time of Filled breaks, feed_backs
	# aligment
	try:
		if args.left:
			parser = spp.sppasRW (transcription_left_palign)
			tier_left_palign = parser.read(). find ("TokensAlign")
			tier_align = tier_left_palign
		else:
			parser = spp.sppasRW (transcription_right_palign)
			tier_right_palign = parser.read(). find ("TokensAlign")
			tier_align = tier_right_palign
		# Time of Filled breaks, feed_backs
		filled_breaks = ts. get_durations (tier_align, list_of_tokens =  FILLED_PAUSE_ITEMS)
		main_feed_items = ts. get_durations (tier_align, list_of_tokens =  MAIN_FEEDBACK_ITEMS)
		main_discourse_items = ts. get_durations (tier_align, list_of_tokens =  MAIN_DISCOURSE_ITEMS)
		main_particles_items = ts. get_durations (tier_align, list_of_tokens =  MAIN_PARTICLES_ITEMS)
		laughters = ts. get_durations (tier_align, list_of_tokens =  LAUGHTER_FORMS)

		'''filled_breaks = ts. get_ratio (tier, list_of_tokens =  FILLED_PAUSE_ITEMS)
		main_feed_items = ts. get_ratio (tier, list_of_tokens =  MAIN_FEEDBACK_ITEMS)
		main_discourse_items = ts. get_ratio (tier, list_of_tokens =  MAIN_DISCOURSE_ITEMS)
		main_particles_items = ts. get_ratio (tier, list_of_tokens =  MAIN_PARTICLES_ITEMS)'''

	except:
		print ("error in computing intervals")


	# Ratios of Filled breaks, feed_backs
	'''filled_breaks = ts. get_ratio (tier, list_of_tokens =  FILLED_PAUSE_ITEMS)
	main_feed_items = ts. get_ratio (tier, list_of_tokens =  MAIN_FEEDBACK_ITEMS)
	main_discourse_items = ts. get_ratio (tier, list_of_tokens =  MAIN_DISCOURSE_ITEMS)
	main_particles_items = ts. get_ratio (tier, list_of_tokens =  MAIN_PARTICLES_ITEMS)'''

	x_emotions, polarity, subejctivity = ts. emotion_ts_from_text (tier, nlp)

	# Time series dictionary
	#time_series = {"Signal": [signal_x, envelope],
	time_series = {
				"Signal": [signal_x, envelope],
				"talk": talk,
				"IPU": IPU,
				"Overlap": overlap,
				"ReactionTime":reaction_time,
				"FilledBreaks":filled_breaks,
				"Feedbacks":main_feed_items,
				"Discourses":main_discourse_items,
				"Particles":main_particles_items,
				"Laughters":laughters,
				"LexicalRichness1":richess_lex1,
				"LexicalRichness2":richess_lex2,
 				"Polarity": [x_emotions, polarity],
				"Subjectivity": [x_emotions, subejctivity],
				}

	labels = ["Signal", "talk", "IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks", "Discourses",
				"Particles", "Laughters", "LexicalRichness1", "LexicalRichness2", "Polarity", "Subjectivity"]

	markers = ['' for i in range (len (labels))]

	# Align time series for visualisation
	#for label in labels[1:]:
		#time_series[label] = ts. align_ts (time_series[label], [0, 60])
	#exit (1)
	## test sampling
	df = pd.DataFrame ()

	# Conbstruct a dataframe with smapled time serie according to the physio index
	df["Time (s)"] = physio_index
	#df ["Silence"] = silence
	for label in labels [:]:
		if label in ["talk"]:
			df [label] = talk [1]
		elif (label in ["IPU", "Overlap", "FilledBreaks", "Laughters", "Feedbacks", "Discourses","Particles"]):
			df [label] = new_sample_square (time_series [label], physio_index)
		else:
			df [label] = sample_cont_ts (time_series [label], physio_index)

	if args.left:
		for i in range (len (labels)):
			labels [i] += "_left"
	df. columns = ["Time (s)"] + labels
	# Output files
	df.to_pickle(output_filename_pkl)
	#ts. plot_time_series ([time_series[label] for label in labels], labels, colors[0:len (labels)], markers=markers,figsize=(20,16), figname = output_filename_1)
