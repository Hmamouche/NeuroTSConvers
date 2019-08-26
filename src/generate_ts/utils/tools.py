import spacy as sp
import numpy as np
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import matplotlib. pyplot as plt


#-----------------------------------------------------------------------------
# Plot dataframe
def plot_df (df, labels, figname, figsize=(12,9), y_lim = [0,1]):
	fig = plt.figure(figsize = figsize)

	plt.title('Title!', color='black')
	df. plot (y=labels, sharex=True, subplots=True, ylim = y_lim, sharey = True, grid=True, legend= False, ax=fig.gca())

	ld = fig.legend(labels = labels,
           loc='upper right',   # Position of legend
           prop={'size':6},
           ncol=1,
           fontsize = 'small')

    #plot_df (squared_ts_sampled, labels_s, colors, markers, figsize=(10,6), figname = "fig.pdf", plot_types = 0)
	plt. savefig (figname, additional_artists = [ld], bbox_inches='tight')
	plt. cla ()
	plt. close ()

#--------------------------------------------------------------------------------------------------
# Plot a list of time series with a list if labels, colors, markers, ...
def plot_time_series (time_series, labels_, colors, markers, figsize=(10,6), figname = "fig.pdf"):

	plt.rcParams['axes.facecolor'] = 'white'
	plt. rc('text', usetex=True)

	fig, ax = plt.subplots (len (labels_), 1, figsize=(10,6))#, sharex=True, sharey=False)

	i = 0
	for (ts_, label, color, marker) in zip (time_series, labels_, colors, markers):

		if len (ts_) == 1:
			ax[i]. plot (ts_[0], marker = marker, label = label, color = color, linewidth = 1)[0]
			ax[i]. grid (True)

		elif len (ts_) == 2:
			ax[i]. plot (ts_[0], ts_[1], marker = marker, color = color, linewidth = 1)[0]
			ax[i]. grid (True)

		ax[i].spines['top'].set_visible(False)
		ax[i].spines['right'].set_visible(False)

		ax[i].tick_params (axis = 'both', labelsize=7)
		i += 1

	fig.text(0.5, 0.04, 'Time', ha='center')
	fig.text(0.04, 0.5, 'Series', va='center', rotation='vertical')

	# Create the legend
	ld = fig.legend (labels = labels_,
           loc='upper right',   # Position of legend
           prop={'size':6},
           ncol=1,
           fontsize = 'small')

	## PLOTS CONFIG
	plt. subplots_adjust(left=None, bottom=None, right=0.8, top=None, wspace=None, hspace=0.9)

	## Save files
	plt.ylabel('', fontsize=10)
	plt.xlabel('', fontsize=10)

	#plt. show ()
	plt.savefig (figname, additional_artists = [ld], bbox_inches='tight')
	plt. cla ()
	plt. close ()

#--------------------------------------------------------------
def get_interval (sppasObject):
	label = sppasObject. serialize_labels()
	location = sppasObject. get_location ()[0][0]

	start = location. get_begin (). get_midpoint ()
	stop = location. get_end (). get_midpoint ()

	start_radius = location. get_begin (). get_radius ()
	stop_radius = location. get_end (). get_radius ()

	# the radius represents the uncertainty of the localization
	return label, [start, stop], [start_radius, stop_radius]

#========================================================

def normalize (signal):
	M = signal
	M. setflags(write=1)
	max = np.max(M)
	min = np.min(M)
	for i in range (len (M)):
		M[i] = (M[i] - min) / (max - min)

	return M

#========================================================

def get_dicretized_ipu (tier, ax, value):
	x = []
	y = [0 for i in range (len (ax))]

	# TODO : OPTIMIZE this loop
	for sppasOb  in tier:
		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)

		for j in range (len (ax)):
			if ax[j] <= stop and ax[j] >= start:
				if label in ["#", "", " ", "***", "*"] :
					y[j] = 0
				else:
					y[j] = 1
	return y

#=========================================================

def get_ipu (tier, value):
	x = []
	y = []
	#tier = tg.tierDict[item]. entryList

	for sppasOb  in tier:

		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)
		x. append (start)
		x. append (stop);

		#x. append ([start, stop])

		if label in ["#", "", " ", "***", "*"] :
			y. append (0)
			y. append (0)
		else:
			y. append (value)
			y. append (value)

		if y[-1] != 0:
			y.append (0)
			x. append (x[-1])

	return x, y

#===================================================
def new_get_ipu (tier, value):
	x = []
	y = []
	#tier = tg.tierDict[item]. entryList

	for sppasOb  in tier:

		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)
		'''x. append (start)
		x. append (stop);
		x. append ([start, stop])'''

		if label in ["#", "", " ", "***", "*"] :
			continue

		else:
			x. append ([start, stop])
			y. append (value)

	return x, y

#=========================================================

def get_item_ts (tier, value = 1):
	x = []
	y = []
	#tier = tg.tierDict[item]. entryList

	for sppasOb  in tier:

		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)
		x. append (start)
		x. append (stop);

		if label in ["#", "", " ", "***", "*"] :
			y. append (0)
			y. append (0)
		else:
			y. append (value)
			y. append (value)

		if y[-1] != 0:
			y.append (0)
			x. append (x[-1])

	return x, y
#=====================================================================
# Get overlap
def get_overlap (tier_left, tier_right, value = 1.0):

	x = []
	y = []


	for sppasOb_l  in tier_left:

		label_l, [start_l, stop_l], [radius_l, radius_l] = get_interval (sppasOb_l)

		if label_l in [""," ", "#","***"]:
			continue

		for sppasOb_r  in tier_right:

			label_r, [start_r, stop_r], [radius_r, radius_r] = get_interval (sppasOb_r)

			if label_r not in ["", "#"," ", "***"] and max (start_l, start_r) < min (stop_l, stop_r):
				x. append (max (start_l, start_r))
				x. append (max (start_l, start_r)+0.0000001)
				x. append (min (stop_l, stop_r))
				x. append (min (stop_l, stop_r)+ 0.0000001)
				y. append (0.0)
				y. append (value)
				y. append (value)
				y. append (0.0)

	if len (x) == 0:
		return [0, 60], [0, 0]

	if y[-1] != 0:
		y.append (0)
		x. append (x[-1])


	if x[-1] != 60:
		y.append (0)
		x. append (60)

	return x, y

#=====================================================
def get_overlap_new (tier_left, tier_right, value = 1.0):

	x = []
	y = []

	for sppasOb_l  in tier_left:

		label_l, [start_l, stop_l], [radius_l, radius_l] = get_interval (sppasOb_l)

		if label_l in [u'+',u'#',u'',u' ', u'*']:
			continue

		for sppasOb_r  in tier_right:

			label_r, [start_r, stop_r], [radius_r, radius_r] = get_interval (sppasOb_r)

			if label_r not in ["", "#"," ", "***"] and max (start_l, start_r) < min (stop_l, stop_r):
				x. append ([max (start_l, start_r), min (stop_l, stop_r)])
				#y. append (value)

	return x
#======================================================================
# Get reaction time
def get_reaction_time (tier_left, tier_right):

	x = []
	y = []

	# Store left intervals and the successor right interval
	intervs_reponses = []


	for sppasOb_l  in tier_left:

		label_l, [start_l, stop_l], [radius_l, radius_l] = get_interval (sppasOb_l)

		if label_l in ["", "#", "***", "*"]:
			continue

		for sppasOb_r  in tier_right:

			label_r, [start_r, stop_r], [radius_r, radius_r] = get_interval (sppasOb_r)

			if label_r in ["", "#", "***", "*"]:
				continue

			if start_l < start_r:
				intervs_reponses. append ([[start_l, stop_l], [start_r, stop_r]])
				break

	# Eliminate doulbed successors: keep just the closed interval
	for i in range (len (intervs_reponses) - 1):
		if intervs_reponses[i][1] == intervs_reponses[i+1][1]:
			intervs_reponses[i][1] = []

	# Computing the reaction time from the begining of the end of the first interval
	for interv in intervs_reponses:
		if (len (interv[1]) > 0):
			x. append (interv[0][1])
			y. append (interv[1][0] - interv[0][1])


	if len (x) == 0:
		return [0, 60], [0, 0]

	return x, y

# Compute richess_lexicale with 2 methods
# method 1 : number of adj + number of adv / total number of tokens
# method 2 : number of different tokes / total
def richess_lexicale (phrase, nlp,  method = "meth1"):

	doc = nlp(phrase)

	if method == "meth1":
		nb_adj = 0
		nb_adv = 0
		total_tokens = 0

		for token in doc:
			if token.pos_ != "PUNCT":
				total_tokens += 1
			if token.pos_ == "ADJ":
				nb_adj += 1

			if token.pos_ == "ADV":
				nb_adv += 1

		if total_tokens == 0:
			return 0

		return float (nb_adv + nb_adj) / total_tokens

	elif method == "meth2":
		token_without_punct = []

		for token in doc:
			#if token. IS_PUNCT == 0:
			if token.pos_ != "PUNCT":
				token_without_punct. append (token.pos_)

		# List of different tokens
		different_tokens = set (token_without_punct)

		if len (token_without_punct) == 0:
			return 0

		return float (len (different_tokens)) / len (token_without_punct)

	else:
		print ("Error, the methode name is no correct, chose 'methd1' or 'meth2'")
		exit (1)

# Generate time series  (two vectors) corresponding to the richness text (using previous function)
def generate_RL_ts (tier, nlp, method = "meth1"):
	x = []
	y = []

	for sppasOb  in tier:

		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)

		x. append (stop)
		if label in ["#", "", " ", "***", "*"] :
			y. append (0)
		else:
			y. append (richess_lexicale (label, nlp = nlp, method = method))
	return x, y


# Get emotions from text : polarity and subjectivity index
# polarity in [-1, 1] : -1 pessimiste, 1 optimiste
# Sunjectivity in [0, 1]
def emotion_ts_from_text (tier, nlp):
	x = []
	y = []
	z = []

	tb = Blobber(pos_tagger = PatternTagger(), analyzer = PatternAnalyzer())
	
	for sppasOb  in tier:

		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)

		x. append (stop)
		if label in ["#", "", " ", "***", "*"] :
			y. append (0)
			z. append (0)
		else:
			bob = tb (label)
			polarity_and_sunjectivity = bob.sentiment
			y. append (polarity_and_sunjectivity[0])
			z. append (polarity_and_sunjectivity[1])
	return x, y, z



#-------------------------------------
def get_sub_times (tier, list_of_tokens):

	x = []
	y = []

	for sppasOb  in tier:

		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)
		x. append (start)
		x. append (stop)

		if label in list_of_tokens:
			y. append (1)
			y. append (1)

		else:
			y. append (0)
			y. append (0)

	if y[-1] != 0:
		y.append (0)
		x. append (x[-1])

	return x, y

#====================================================
def get_durations (tier, list_of_tokens):

	x = []
	y = []

	for sppasOb  in tier:

		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)

		if label in list_of_tokens:
			x. append ([start, stop])

	return x

#---------------------------------------------
# Get ratios of items like discourse markers,
def get_ratio (tier, list_of_tokens):
	total_number_of_tokens = 0
	ratio_of_tokens = []
	x = [] # abcisse

	for sppasOb  in tier:

		number_of_tokens_in_label = 0
		label, [start, stop], [start_r, stop_r] = get_interval (sppasOb)

		if label in ["#", "", " ", "***", "*"]:
			continue

		total_number_of_tokens += 1

		for token in list_of_tokens:
				number_of_tokens_in_label += label.lower ().count(token. lower ())

		ratio_of_tokens. append (number_of_tokens_in_label)
		x. append (float (start + stop) / 2)

	if total_number_of_tokens == 0:
		return [], []

	for i in range (len (ratio_of_tokens)):
		ratio_of_tokens [i] = float (ratio_of_tokens [i]) / float (total_number_of_tokens)

	return x, ratio_of_tokens





# Make time series start from 0 and end in 60
def align_ts (ts, interv):
	if len (ts [0]) == 0:
		return ts
	x = ts [0][:]
	y = ts [1][:]

	if x[0] != interv[0]:
		x = [interv[0]] + x
		y = [0] + y

	if x[-1] != interv[1]:
		x. append (interv[1])
		y. append (0.0)

	return [x, y]
