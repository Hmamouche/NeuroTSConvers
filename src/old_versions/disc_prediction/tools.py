import sys
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.ensemble import RandomForestClassifier


stderr = sys.stderr
sys.stderr = open(os.devnull, 'w') # hide keras messages
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras import regularizers
sys.stderr = stderr

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

sh_models = {
    "BayeRidge":linear_model.BayesianRidge(),
    "Lasso1":linear_model.Lasso(alpha = 0.01),
    "Ridge1":linear_model.Ridge(alpha = 0.0),
    "LinearRegression":linear_model. LinearRegression ()
}

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

def mlp_regressor (in_dim, out_dim, start_weigths, set_coefs = False):
	# define our MLP network

	nb_neurons = int ((in_dim + out_dim ) / 2)
	model = Sequential()
	model.add (Dense(1, kernel_initializer='normal', input_dim=in_dim))
	#model.add(Dropout(0.2))
	#model. add (Activation ("relu"))

	#model.add (Dense(5))
	#model. add (Activation ("relu"))

	#model.add (Dense(5))
	#model. add (Activation ("relu"))

	#model.add (Dense(2))
	#model. add (Activation ("relu"))

	model.add(Dense(1))
	model. add (Activation ("sigmoid"))

	if set_coefs:
		model. layers [0]. set_weights ([start_weigths, np.array ([0.0 for i in range (out_dim)]) ] )

	# return our model
	return model

#----------------------------------------------------------------------------------------------------
def  train_global_model (convers, target_column, target_index, subject, predictors, external_predictors, lag, epochs, batch_size=5, verbose=0, add_target = True):

	filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, convers[0])

	if external_predictors != None:
		external_data = get_behavioral_data (subject, convers[0], external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]

	# normalize data
	scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data =  scaler. transform (data)

	#print (data)

	# Charge keras layer
	supervised_data = toSuppervisedData (data, lag, add_target = add_target)
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]

	model = mlp_regressor (X.shape[1], 1, start_weigths = None)
	model. compile (optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
	model.fit (X, Y, epochs = epochs, batch_size = batch_size, verbose = verbose)

	#fit_var_mlp (data, target_index, lag, model, normalize = True, epochs = epochs, batch_size = batch_size, verbose=verbose)

	# Online model updating on  the rest of conversations
	for conv in convers[1:]:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		if external_predictors != None:
			external_data = get_behavioral_data (subject, conv, external_predictors)
			data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
		else:
			data = pd.read_pickle (filename)[[target_column] + predictors]. values

		fit_var_mlp (data, target_index, lag, model, normalize = True, epochs = epochs, batch_size = batch_size, verbose=verbose, add_target = add_target)

	return model

#----------------------------------------------------------------------------------------------------
def  test_global_model (model, conv, target_index, target_column, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0, lag_max=5, add_target = True):
	results = []

	filename = "time_series/%s/discretized_physio_ts/%s.pkl" %(subject, conv)

	if not os.path.exists (filename):
		print ("file does not exist")

	if external_predictors != None:
		external_data = get_behavioral_data (subject, conv, external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]. values

	scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data = scaler. transform (data)

	real = data [lag_max:,target_index]
	start_points = data [0:lag,:]
	pred = []

	for i in range (lag_max, data. shape [0]):

		supervised_data = toSuppervisedData (data[i-lag:i+1,:], lag, add_target = add_target)
		X = supervised_data.data
		Y = supervised_data.targets [:,target_index]
		# Make one prediction
		#predictions = predict_var_mlp (data[i-lag-1:i,:], target_index, lag, model, normalize = False, batch_size=1)
		predictions = model. predict (X, batch_size = batch_size)
		print (predictions[-1][0])
		if (predictions[-1][0] > 0.4):
			pred. append (1.0)
		else:
			pred. append (0.0)

		# Update the model
		model. fit (X, Y, verbose = verbose, epochs=epochs, batch_size = batch_size)

	print (real)
	print (pred)

	score = [recall_score (real, pred), precision_score (real, pred)]

	print (score)

	exit (1)

	return score


#----------------------------------------------------------------------------------------------------
def fit_var_mlp (data, target_index, lag, model, normalize = False, epochs=10, batch_size=1, verbose=0, add_target = True) :

	# normalize data
	if normalize:
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)

	supervised_data = toSuppervisedData (data, lag, add_target = add_target)
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]
	model.fit (X, Y, epochs = epochs, batch_size = batch_size, verbose = verbose)

#----- Fit and train random forrest
def  train_random_forrest (convers, target_column, target_index, subject, predictors, external_predictors, lag, add_target = True):

	X_all = np.empty ([0, 0])
	Y_all = np.empty ([0])
	# Online model updating on  the rest of conversations
	for conv in convers:
		filename = "time_series/%s/discretized_physio_ts/%s.pkl"%(subject, conv)

		if external_predictors != None:
			external_data = get_behavioral_data (subject, conv, external_predictors)
			data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
		else:
			data = pd.read_pickle (filename)[[target_column] + predictors]. values

		# normalize data
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)

		supervised_data = toSuppervisedData (data, lag, add_target = add_target)
		X = supervised_data.data
		Y = supervised_data.targets [:,target_index]

		if X_all. size == 0:
			X_all = X
			Y_all = Y
		else:
			X_all = np. concatenate ((X_all,X), axis = 0)
			Y_all = np. concatenate ((Y_all, Y), axis = 0)

	#X, Y = make_classification (n_samples = X. shape[0], n_features = X. shape[1], n_classes = 2)
	clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
	clf. fit (X, Y)

	return clf
#-----------------------------------------------------------------------
def  test_random_forest (model, conv, target_index, target_column, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0, lag_max=5, add_target = True):
	results = []

	filename = "time_series/%s/discretized_physio_ts/%s.pkl" %(subject, conv)

	if not os.path.exists (filename):
		print ("file does not exist")

	if external_predictors != None:
		external_data = get_behavioral_data (subject, conv, external_predictors)
		# concat physio and behavioral data
		data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
	else:
		data = pd.read_pickle (filename)[[target_column] + predictors]. values

	scaler = MinMaxScaler()
	scaler.fit_transform(data)
	data = scaler. transform (data)

	supervised_data = toSuppervisedData (data, lag, add_target = add_target)
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]

	pred = model. predict (X)
	real = Y

	score = [recall_score (real, pred), precision_score (real, pred), f1_score (real, pred)]

	return score
