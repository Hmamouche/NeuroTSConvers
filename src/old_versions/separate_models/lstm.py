# -*- coding: utf-8 -*-
# Author: Youssef Hmamouche

import sys
import os
import argparse

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w') # hide keras messages
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras import regularizers
sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--------------------------------------------------------

def fit_lstm (data, target_index, lag, model, build = True, epochs=10, batch_size=1, verbose=0) :

 	X = data.reshape(X.shape[0], lag, X.shape[1] / look_back)
	if build:
		model = Sequential()
		model.add (LSTM (lag, batch_input_shape = (batch_size, X.shape[1], X.shape[2])))
		model.add (Dense (1))
		model. add (Activation ("sigmoid"))

		#model.compile (loss='mean_squared_error', optimizer='adam')
		model. compile (optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

	for i in range(nb_epochs):
		model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False, Stateful=False)


def lstm_predict (model, data, lag):
    X = data.reshape(X.shape[0], lag, X.shape[1] / lag)
    return model. predict (X)



#--------------------------------------------------------
