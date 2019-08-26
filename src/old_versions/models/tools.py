import sys
import os

import numpy as np
import pandas as pd

def write_line (filename, row, colnames, mode = "a+"):

	# Write data to filename line by line
	# Data is supposed to be a list of lists

	if not os.path.exists (filename):
		f=open (filename, "w+")
		f.write (';'. join (colnames))
		f. write ('\n')

		for i in range (len (row)):
			row[i] = str (row[i])
		f.write (';'. join (row))
		f. write ('\n')
		f.close ()
	else:
		f=open(filename, mode)
		for i in range (len (row)):
			row[i] = str (row[i])
		f.write (';'. join (row))
		f. write ('\n')
		f. close ()
	return

def write_lines (filename, data, colnames, mode = "a+"):

	# Write data to filename line by line
	# Data is supposed to be a list of lists

	if not os.path.exists (filename):
		f=open (filename, "w+")
		f.write (','. join (colnames))
		f. write ('\n')
		for row in data:
			for i in range (len (row)):
				row[i] = str (row[i])
			f.write (','. join (row))
			f. write ('\n')
		f.close ()
	else:
		f=open(filename, mode)
		for row in data:
			for i in range (len (row)):
				row[i] = str (row[i])
			f.write (';'. join (row))
			f. write ('\n')
		f. close ()
	return

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

#------------------------------------------

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
#------------------------------------------

class toSuppervisedData:
	targets = np.empty([0,0],dtype=float)
	data = np.empty([0,0],dtype=float)

	## constructor
	def __init__(self, X, p, test_data = False, add_target = True):

		self.targets = self.targets_decomposition (X,p)
		self.data = self.matrix_decomposition (X,p, test_data)

		if not add_target:
			self.data = np. delete (self.data, range (0,p), axis = 1)

	## p-decomposition of a vector
	def vector_decomposition (self,x, p, test = False):
		n = len(x)
		if test:
			add_target_to_data = 1
		else:
			add_target_to_data = 0

		output = np.empty([n-p,p],dtype=float)

		for i in range (n-p):
			for j in range (p):
				output[i,j] = x[i + j + add_target_to_data]

		output = np. mean (output, axis = 1)
		output = np. reshape (output, [n-p, 1])
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

#----------------------------------
