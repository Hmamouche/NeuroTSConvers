import sys
import os

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

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
	def __init__(self, X, p, test_data = False):
			
		self.targets = self.targets_decomposition (X,p)
		self.data = self.matrix_decomposition (X,p, test_data)

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

#------------------------------------------

def var_model (data, p, model_ = "LinearRegression", target_index = [0]):

	# df : dataframe
	# p : the lag parameter
	
	supervised_data = toSuppervisedData (data, p)
	model = sh_models [model_]
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]
	model = model.fit (X, Y)
	return X, Y, model


#----------------------------------

def mlp_regressor (in_dim, out_dim, start_weigths, set_coefs = True):
	# define our MLP network
	model = Sequential()
	model.add (Dense(1, input_dim=in_dim))
	model. add (Activation ("relu"))
	
	#model.add(Dropout(0.15))
	
	#model.add (Dense(out_dim))
	#model. add (Activation("relu"))
	
	if set_coefs:
		model. layers [0]. set_weights ([start_weigths, np.array ([0.0 for i in range (out_dim)]) ] )
	
	# return our model
	return model

#----------------------------------------------------------------------------------------------------				
def  train_global_model (convers, target_column, target_index, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0):
	# convers: list of conversations file names
	# target_index
	# sunject
	# predictors: the physio variables names to add in the model
	# lag : the lag parameter
	# Read first filename
	
	filename = "time_series/%s/physio_ts/%s.pkl"%(subject, convers[0])
	
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
	
	# Linear VAR model
	X, Y, model = var_model (data, p=lag, target_index = [target_index])
	weights = np.reshape (model.coef_, (X. shape[1], 1))
	
	# Charge keras layer
	model = mlp_regressor (X.shape[1], 1, start_weigths = weights)
	model. compile(loss='mean_squared_error', optimizer='SGD')
	#model. fit (X, Y, verbose = verbose, epochs=epochs, batch_size = batch_size)
	
	# Online model updating on  the rest of conversations
	for conv in convers[1:]:
		filename = "time_series/%s/physio_ts/%s.pkl"%(subject, conv)
		
		if external_predictors != None:
			external_data = get_behavioral_data (subject, conv, external_predictors)
			data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
		else:
			pd.read_pickle (filename)[[target_column] + predictors]
			
		fit_var_mlp (data, target_index, lag, model, normalize = True, epochs = epochs, batch_size = batch_size, verbose=verbose)
		
	return model

#----------------------------------------------------------------------------------------------------
def  test_global_model (model, convers, target_index, target_column, subject, predictors, external_predictors, lag, epochs=5, batch_size=1, verbose=0, lag_max=5):
	results = []
	for conv in convers:
		filename = "time_series/%s/physio_ts/%s.pkl" %(subject, conv)
		
		if not os.path.exists (filename):
			print ("file does not exist")
			
		if external_predictors != None:
			external_data = get_behavioral_data (subject, convers[0], external_predictors)
			# concat physio and behavioral data
			data = pd. concat ([ pd.read_pickle (filename)[[target_column] + predictors], external_data], axis = 1). values
		else:
			data = pd.read_pickle (filename)[[target_column] + predictors]
		
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)
		
		real = data [lag_max:,target_index]
		start_points = data [0:lag,:]
		pred = []
		#fit_var_mlp (start_points, target_index, lag, model)
		#break
	
		for i in range (lag_max, data. shape [0]):
			
			supervised_data = toSuppervisedData (data[i-lag:i+1,:], lag, False)
			X = supervised_data.data
			Y = supervised_data.targets [:,target_index]
			# Make one prediction
			#predictions = predict_var_mlp (data[i-lag-1:i,:], target_index, lag, model, normalize = False, batch_size=1)
			predictions = model. predict (X, batch_size = batch_size)
			pred. append (predictions[-1][0])
			model. fit (X, Y, verbose = verbose, epochs=epochs, batch_size = batch_size)
			# Update the model
			#fit_var_mlp (data[i-lag:i+1,:], target_index, lag, model, normalize = False, epochs=5, batch_size=1, verbose=0)

		rmse = np. sqrt (mean_squared_error (real, pred))
		
		if external_predictors != None:
			external_variables = []
			for key in external_predictors. keys ():
				external_variables += external_predictors[key]
			variables = '+'.join (predictors + external_variables)
		else:
			variables = '+'.join (predictors)
			
		if int (conv. split ('_')[-1]) % 2 == 1:
			results. append ([subject, target_column,"HH", variables, lag, float (rmse)])
		else:
			results. append ([subject, target_column,"HR", variables, lag, float (rmse)])
			
	return results
	
#----------------------------------------------------------------------------------------------------
def fit_var_mlp (data, target_index, lag, model, normalize = False, epochs=10, batch_size=1, verbose=0) :

	# normalize data
	if normalize:
		scaler = MinMaxScaler()
		scaler.fit_transform(data)
		data =  scaler. transform (data)
		
	supervised_data = toSuppervisedData (data, lag)
	X = supervised_data.data
	Y = supervised_data.targets [:,target_index]
	model.fit (X, Y, epochs = epochs, batch_size = batch_size, verbose = verbose)

