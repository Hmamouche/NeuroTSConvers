import sys
import os

import numpy as np
import pandas as pd
import random as rd

from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

#=======================================================

def write_line (filename, row, mode = "a+", sep = ';'):
	f = open(filename, mode)
	for i in range (len (row)):
		row[i] = str (row[i])
	f.write (sep. join (row))
	f. write ('\n')
	f. close ()
	return

#=======================================================

def list_convers (n_blocks = 4, n_convs = 6):
	# Return the list of conversations names in the format like : CONV2_002
	convs = []
	for t in range (1, n_blocks + 1):
		for j in range (1, n_convs + 1):
			if j%2:
				i = 1
			else:
				i = 2
			convs. append ("convers-TestBlocks%s_CONV%d"%(t, i) +  "_%03d"%j)
	return convs

#=======================================================

def get_behavioral_data (subject, convers, external_predictors):
	external_data = np. array ([])
	external_filenames = ["time_series/%s/%s/%s.pkl"%(subject, data_type, convers) for data_type in external_predictors. keys ()]
	external_columns = [external_predictors[item] for item in external_predictors. keys ()]

	for filename, columns in zip (external_filenames, external_columns):

		if external_data. size == 0:
			if os. path. exists (filename):
				try:
					external_data = pd.read_pickle (filename)[columns]. values
				except:
					continue
		else:
			if os. path. exists (filename):
				try:
					external_data = np. concatenate ((external_data, pd.read_pickle (filename)[columns]. values), axis = 1)
				except:
					continue

	return external_data

#============================================================

def shuffle_data_by_blocks (data, block_size):

	blocks = [data [i : i + block_size] for i in range (0, len(data), block_size)]
	# shuffle the blocks
	rd.shuffle (blocks)
	# concatenate the shuffled blocks
	output = [b for bs in blocks for b in bs]
	output = np. array (output)
	return output


#============================================================

def train_test_split (data, test_size = 0.2):

	nb_obs = int (data. shape [0] * (1 - test_size))
	train_d = data [0 : nb_obs, :]
	test_d = data [nb_obs :, :]

	return train_d, test_d

#============================================================

def get_lagged_colnames (behavioral_predictors, lag):
	# get behavioral variables colnames with time lag
	columns = []
	lagged_columns = []

	for item in behavioral_predictors. keys ():
		items = behavioral_predictors[item]
		columns. extend (items)

	for item in columns:
		lagged_columns. extend ([item + "_t%d"%(p) for p in range (lag, 2, -1)])
		#lagged_columns. extend ([item])

	return (lagged_columns)

#=========================================================================================
def concat_ (subject, target_column, convers, lag, behavioral_predictors, add_target = False, reg = False):

	data = pd. DataFrame ()
	for conv in convers:

		if reg:
			filename = "time_series/%s/physio_ts/%s.pkl"%(subject, conv)
		else:
			filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		target = pd.read_pickle (filename). loc [:,[target_column]]. values

		# Load neurophysio and behavioral predictors
		if len (behavioral_predictors) > 0:
			external_data = get_behavioral_data (subject, conv, behavioral_predictors)
		else:
			external_data = None

		if external_data. shape[0] == 0:
			continue

		# concatenate data of all conversations
		if data.shape [0] == 0:
			data = np. concatenate ((target, external_data), axis = 1)
			if lag > 0:
				supervised_data = toSuppervisedData (data, lag, add_target = add_target)
				data = np.concatenate ((supervised_data. targets[:,0:1], supervised_data.data), axis = 1)

		else:
			#U = pd. concat ([target, external_data], axis = 1)
			U = np. concatenate ((target, external_data), axis = 1)
			if lag > 0:
				supervised_U = toSuppervisedData (U, lag, add_target = add_target)
				U = np.concatenate ((supervised_U. targets[:,0:1], supervised_U.data), axis = 1)
			data =  np. concatenate ((data, U), axis = 0)

		# DEBUG CONVERSATION
		'''if data. shape [1] == 0:
			print (18 * "=")
			print (conv)
			print (subject)
			exit (1)'''

	return data

#=======================================================
class toSuppervisedData:
	targets = np.empty([0,0],dtype=float)
	data = np.empty([0,0],dtype=float)

	## constructor
	def __init__(self, X, p, test_data = False, add_target = False):

		self.targets = self.targets_decomposition (X,p)
		self.data = self.matrix_decomposition (X,p, test_data)

		if not add_target:
			self.data = np. delete (self.data, range (0, p), axis = 1)

		delet = []

		if X.shape[1] > 1 and p > 4:
			for j in range (0, self.data. shape [1], p):
				delet. extend ([j + p - i for i in range (1, 3)])

		self.data = np. delete (self.data, delet, axis = 1)

		# compute the mean of lagged variables
		'''n_var = int (self.data. shape [1] / p)
		new_data = np.empty ([self.data. shape [0], 2 * n_var], dtype = np. float64)
		new_data = np.array ([])

		if X.shape[1] > 2:
			for j in range (0, self.data. shape [1], p):
				# [0, 1, 2] is equivalent to t-3, t-4, t-5
				col = self.data [:, [j + i for i in range (4)]]
				#new_data [:, int (j / p)] = np. mean (col, axis = 1)

				# make a PCA on each 	lagged variables
				model = PCA (n_components = 3)
				factors = model.fit_transform (col)

				if (np.isnan (factors). any ()):
					print ("KAYEN")

				if new_data. shape [0] == 0:
					new_data = factors
				else:
					new_data = np. concatenate ((new_data, factors), axis = 1)

			self.data = new_data'''

	## p-decomposition of a vector
	def vector_decomposition (self, x, p, test = False):
		n = len(x)
		if test:
			add_target_to_data = 1
		else:
			add_target_to_data = 0

		output = np.empty([n-p,p],dtype=float)

		for i in range (n-p):
			for j in range (p):
				output[i,j] = x[i + j + add_target_to_data]
		return output

	# p-decomposition of a target
	def target_decomposition (self,x,p):
		n = len(x)
		output = np.empty([n-p,1],dtype=float)
		for i in range (n-p):
			output[i] = x[i+p]
		return output

	# p-decomposition of a matrix
	def matrix_decomposition (self,x,p, test=False):
		output = np.empty([0,0],dtype=float)
		out = np.empty([0,0],dtype=float)

		for i in range(x.shape[1]):
			out = self.vector_decomposition(x[:,i],p, test)
			if output.size == 0:
				output = out
			else:
				output = np.concatenate ((output,out),axis=1)

		return output
	# extract all the targets decomposed
	def targets_decomposition (self,x,p):
		output = np.empty([0,0],dtype=float)
		out = np.empty([0,0],dtype=float)
		for i in range(x.shape[1]):
			out = self.target_decomposition(x[:,i],p)
			if output.size == 0:
				output = out
			else:
				output = np.concatenate ((output,out),axis=1)
		return output
