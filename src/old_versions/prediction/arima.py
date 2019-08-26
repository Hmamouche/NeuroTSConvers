import pandas as pd
#from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA

import argparse
import sys
import os
from glob import glob
import numpy as np

from sklearn.metrics import mean_squared_error

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
	
def get_filename (subject, testblock, conv, conv_numb):
	subject = "sub-%02d"%subject
	conv_numb = "%03d"%conv_numb
	return "time_series/%s/physio_ts/convers-TestBlocks%s_CONV%s_%s.pkl"%(subject, testblock, conv, conv_numb)


#----------------------------
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


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("subject", type=int)
	parser.add_argument("--write", "-w", help="write results", action="store_true")
	
	args = parser.parse_args()
	
	lag_max = 5
	subject = "sub-%02d"%args.subject
	
	filename_hh = "results/sub-%02d/predictions_arima_HH.txt"%args.subject
	filename_hr = "results/sub-%02d/predictions_arima_HR.txt"%args.subject
	
	os. system ("rm %s"%filename_hh)
	os. system ("rm %s"%filename_hr)

	
	conversatsions = list_convers ()

	first_file = True
	
	regions = ["region_73", "region_74", "region_75", "region_76", "region_79","region_80","region_87","region_88", "region_121", 
	"region_122", "region_123", "region_124"]
	colnames = ["subject", "Region", "Rmse"]

	for target_column in regions:
	# Train the mode	
		for conv in conversatsions[:-2]:
		
			filename = "time_series/%s/physio_ts/%s.pkl"%(subject, conv)
			data = pd.read_pickle (filename)
			
			if first_file:
				model = auto_arima (data[target_column], seasonal = False)
				#print (model. params ())
				first_file = False
			else:
				model. update (data[target_column], maxiter=30)
				
		# Test the model on the last two conversations
		results = []
		for conv in conversatsions[-2:]:
			filename = "time_series/%s/physio_ts/%s.pkl"%(subject, conv)
			data = pd.read_pickle (filename)
			
			y = data. loc[:, target_column]. values
			start_points = y[0:lag_max]
			real = y [lag_max:]
			pred = []
			model. update (start_points, maxiter=5)
			
			for i in range (lag_max, data. shape [0]):
				pred. append (model.predict (1)[0])
				model. update (y[:i+1])
				
			rmse = np. sqrt (mean_squared_error (real, pred))
			
			if args. write:
				if int (conv. split ('_')[-1]) % 2 == 1:
					write_line (filename_hh, [subject, target_column, rmse], colnames, mode="a+")
				else:
					write_line (filename_hr, [subject, target_column, rmse], colnames, mode="a+")

		






	
	
	
	
	
	
	
	
	





