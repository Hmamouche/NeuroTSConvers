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
from keras.utils import to_categorical
from keras.utils import to_categorical

sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--------------------------------------------------------

def fit_lstm (X, Y, lag, params) :

    batch_size = 1
    #read Parameters
    nb_epochs = params ['epochs']
    nb_neurons = params ['neurons']

    #X_reshaped = X.reshape(X.shape[0], lag, int (X.shape[1] / lag))

    model = Sequential()
    #model.add (LSTM (units = 16, stateful=True,  batch_input_shape = (batch_size, X_reshaped.shape[1], X_reshaped.shape[2]), kernel_initializer='random_uniform'))

    nb_neurons = int (float (X.shape [1] + 1) * 0.667)
    model.add (Dense (nb_neurons, input_shape = (X.shape[1],)))
    model. add (Activation ("relu"))

    if (nb_neurons / 2) > 2:
        model.add (Dense (nb_neurons / 2))
        model. add (Activation ("relu"))

    model.add (Dense (1))
    model. add (Activation ("sigmoid"))

    model.compile (loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    model.fit (X, Y,  epochs = 60, batch_size = batch_size, verbose = 0, shuffle = False)

    return model


def lstm_predict (X, model, lag):
    #X_reshaped = X.reshape(X.shape[0], lag, int (X.shape[1] / lag))
    preds = model. predict (X, batch_size = 1). reshape (-1)
    for i in range (len (preds)):
        if preds [i] < 0.5:
            preds [i] = 0
        else:
            preds [i] = 1
    return preds



#--------------------------------------------------------
