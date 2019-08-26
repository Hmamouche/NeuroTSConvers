import sys
import os

import numpy as np
import pandas as pd
import random as rd

#=======================================================

def write_line (filename, row, mode = "a+"):
	f = open(filename, mode)
	for i in range (len (row)):
		row[i] = str (row[i])
	f.write (';'. join (row))
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
	external_data = pd.DataFrame ()
	external_filenames = ["time_series/%s/%s/%s.pkl"%(subject, data_type, convers) for data_type in external_predictors. keys ()]
	external_columns = [external_predictors[item] for item in external_predictors. keys ()]

	for filename, columns in zip (external_filenames, external_columns):

		if external_data. empty:
			external_data = pd.read_pickle (filename)[columns]
		else:
			external_data = pd. concat ([external_data, pd.read_pickle (filename)[columns]], axis = 1)

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

def get_lagged_colnames (behavioral_predictors):
	# get behavioral variables colnames with time lag
	columns = []
	lagged_columns = []

	for item in behavioral_predictors. keys ():
		items = behavioral_predictors[item]
		if "left" in item:
			columns. extend (colname + "_left" for  colname in items)
		else:
			columns. extend (items)

	for item in columns:
		#lagged_columns. extend ([item + "_t5", item + "_t4", item + "_t3"])
		lagged_columns. extend ([item])

	return (lagged_columns)


#=========================================================================================
def concat_ (subject, target_column, convers, lag, behavioral_predictors, add_target = False):
	data = pd. DataFrame ()

	for conv in convers:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		target = pd.read_pickle (filename)[target_column]
		# Load neurophysio and behavioral predictors
		if len (behavioral_predictors) > 0:
			external_data = get_behavioral_data (subject, conv, behavioral_predictors)
		else:
			external_data = None

		# concatenate data of all conversations
		if data.shape [0] == 0:
			data = pd. concat ([target, external_data], axis = 1)
			supervised_data = toSuppervisedData (data.values, lag, add_target = add_target)
			data = np.concatenate ((supervised_data. targets[:,0:1], supervised_data.data), axis = 1)
		else:
			U = pd. concat ([target, external_data], axis = 1)
			supervised_U = toSuppervisedData (U.values, lag, add_target = add_target)
			U = np.concatenate ((supervised_U. targets[:,0:1], supervised_U.data), axis = 1)
			data =  np. concatenate ((data, U), axis = 0)


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
			# self.data = np. delete (self.data, range (0, p), axis = 1)


		delet = []

		'''if X.shape[1] > 1 and p > 4:
			for j in range (0, self.data. shape [1], p):
				#delet. append (j + 0)
				#delet. append (j + 2)
				delet. append (j + 3)
				delet. append (j + 4)'''

		n_var = int (self.data. shape [1] / p)
		delet = np.empty ([self.data. shape [0], n_var], dtype = float)

		if X.shape[1] > 1 and p > 4:
			for j in range (0, self.data. shape [1], p):
				# [0, 1, 2] is equivalent to t-3, t-4, t-5
				col = self.data[:, [j + i for i in range (p)]]
				delet [:, int (j / p)] = np. mean (col, axis = 1)

		self.data = delet
		'''print (self.data. shape)
		print (delet. shape)
		exit (1)'''

			#self.data = np. delete (self.data, delet, axis = 1)

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

		'''output = np. mean (output, axis = 1)
		output = np. reshape (output, [n-p, 1])'''

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
