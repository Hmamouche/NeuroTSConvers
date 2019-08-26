import pandas as pd
from glob import glob
import os
import argparse
import numpy as np
import pywt

import pylab as plt

from scipy.stats import mode as sc_mode

#===================================================

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

#===================================================

def denoise_signal( x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """

    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )

    #print (coeff)

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1/0.6745) * maddest( coeff[-level] )

    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )

    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )

#===========================================

def read_asci (filename):
    f = open(filename, 'r')
    line = f.readline()
    comment = []
    data = []

    while line:
        if line. startswith("**"):
            comment. append ([line])

        else: data. append (line. split ('\n')[0]. split ('\t'))

        line = f.readline()
    f.close()
    return data, comment

#===========================================

def find_event (data, events):
    mess = []
    begin = False

    for line in data:
        if line [0] in events:
            mess. append (line)

    return mess

#===========================================

def is_message (line):
    if line [0] in ["MSG"]: #, "EFIX", "SSACC", "SBLINK", "EBLINK", "ESACC", "SFIX"]:
        return True
    else: return False

#===========================================

def is_R (line):
    if line [0] in ['R']: #, "EFIX", "SSACC", "SBLINK", "EBLINK", "ESACC", "SFIX"]:
        return True
    else: return False

#===========================================

def is_event (line):
    if line [0] in ["EFIX", "SSACC", "SBLINK", "EBLINK", "ESACC", "SFIX"]:
        return True
    else: return False

#===========================================

def to_df (list):
    return pd. DataFrame (list)

#===========================================

# TODO : better organization
def find_start_end (data, start, end):
    start_end = []
    row = []
    no_end = False

    for i in range (len (data)):
        if start in data [i][0]:
            row. append (i)
            no_end = True
        if end in data [i][0]:
            if len (row) == 0:
                row. append (1)
            row. append (i)
            start_end. append (row)
            row = []
            no_end = False

    if no_end:
        row. append (len (data) - 1)
        start_end. append (row)

    return start_end

#=========================================
def find_saccades (data):

    sacc_indices = []
    saccades = find_start_end (data, "SSACC", "ESACC")

    for sacc in saccades:
        sacc_indices. extend (list (range (sacc [0], sacc[1] + 1)))

    return sacc_indices

#==========================================
def find_blinks (data):

    blinks = find_start_end (data, "SBLINK", "EBLINK")
    saccades = find_start_end (data, "SSACC", "ESACC")

    sacc_contain_blinks = []

    for sacc in saccades:
        for blink in blinks:
            if blink [0] >= sacc[0] and blink [1] <= sacc[1]:
                sacc_contain_blinks. append (sacc)
                break

    blinks_indices = []
    blinks. extend (sacc_contain_blinks)

    for blink in blinks:
        blinks_indices. extend (list (range (blink [0], blink[1] + 1)))

    return list (set (blinks_indices))

#===========================================
def find_conv (data, message):
    convers = []
    i = 0
    # Saving conversations points
    for line in data:
        if is_message (line) and message in line [2]:
            convers. append ([line, i, 0, 0])
        i += 1

    for i in range (len (convers) - 1):
        convers[i][2] = convers [i + 1][1] - 1

    convers[-1][2] = len (data) - 1

    for i in range (len (convers)):
        convers[i][3] = convers[i][2] - convers[i][1]

    return convers


#===========================================
def find_convers (data):

    """
        Find the data associated to the 6 conversations, and put
        them into a list of lists. Then removing blinks and saccades.
    """

    convers = find_conv (data, "CONV")
    conversations = [data [conv [1] : conv [2]] for conv in convers]

    for i in range (len (conversations)):
        conv_cleaned = []
        blinks = find_blinks (conversations [i])
        #saccs = find_saccades (conversations [i])

        all = blinks # + saccs

        for j in range (len (conversations [i])):
            if j not in all and conversations [i][j][0] not in ["MSG", "EFIX", "SFIX", "INPUT", "END", "SSACC", "ESACC"]:
                conv_cleaned. append (conversations [i][j])

        conversations [i] = conv_cleaned. copy ()

    return conversations

#==========================================
def mode_with_nan (list_, mode):

    for row in list_:
        if mode == "mean":
            return np.nanmean (list_, axis=0). tolist ()
        elif mode == "max":
            return np.nanmax (list_, axis=0). tolist ()
        elif mode == "mode":
            return sc_mode (list_, axis=0, nan_policy = 'omit')[0][0]. tolist ()


#===========================================

def resample_ts (data, index, mode = "mean"):

    """
        Resampling a time series according to an input index.
        data : a list of observations, representing the input time series.
        the data must contain an index in the first column
        index : the new index (with smaller frequency compared to the data original index)
    """

    resampled_ts = []
    rows_ts = []
    j = 0

    for i in range (len (data)):
        if j >= len (index):
            break

        if (data[i][0] > index [j]):
            resampled_ts. append ([index [j]] + mode_with_nan (rows_ts, mode))
            initializer = 0
            j += 1
            rows_ts = []
        rows_ts. append (data [i][1:])

    if len (rows_ts) > 0 and j < len (index):
        resampled_ts. append ([index [j]] + mode_with_nan (rows_ts, mode))

    return resampled_ts

#==============================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--process", "-p", help = "generate asci files from edf files.", action = "store_true")
    args = parser.parse_args()

    #====================================#
    if args. process:
        edf_files = glob ("data/edt/*.edf*")

        for filename in edf_files:
            os. system ("data/./EDF_Access_API/Example/edf2asc -failsafe -t -miss NaN -y -v %s"%filename)

    asci_files = glob ("data/edt/*.asc*")
    asci_files. sort ()

    #====================================#

    subjects = []
    for i in range(1, 26):
        if i < 10:
            subjects.append("sub-0%s" % str(i))
        else:
            subjects.append("sub-%s" % str(i))

    for subject in subjects:
        if not os.path.exists("time_series/%s/eyetracking_ts" % subject):
            os.makedirs("time_series/%s/eyetracking_ts" % subject)

        if not os.path.exists("time_series/%s/eyes_gradient_ts" % subject):
            os.makedirs("time_series/%s/eyes_gradient_ts" % subject)

    #======================================#

    colnames = ["Time (s)", "x", "y"]

    """ Construct the index: 50 observations in physiological data """
    index = [0.6025]
    for i in range (1, 50):
        index. append (1.205 + index [i - 1])

    """ 1/30: image frequency of videos, equivalent to 1799 images per minute """
    long_index = [1.0 / 30.0 ]
    for i in range (1, 1799):
        long_index. append (1.0 / 30.0 + long_index [i - 1])

    """ process all sci files """
    for filename in asci_files:
        print (filename)

        short_file_name = filename. split ('_')[0:2]
        subject = short_file_name [0]. split ('/')[-1]

        if subject in ["sub-12", "sub-19", "sub-14"]:
            continue

        testBlock = '-'. join (short_file_name [1]. split ('-')[1:3])

        """ get data from asci files """
        data, comment = read_asci (filename)
        convers = find_convers (data)

        """ Extract conversations from concatenated data,
            and initialize the begining time of each conversations at 0 """

        for i in range (0, len (convers)):

            if len (convers [i]) == 0:
                continue

            begin = int (convers[i][0][0])
            for j in range (len (convers[i])):
                convers[i][j][0] = (int (convers[i][j][0]) - begin) / 1000
                for k in range (1,4):
                    convers[i][j][k] = float (convers[i][j][k])

            if i % 2 == 0:
                conv = "CONV1_%03d"%(i+1)
            else:
                conv = "CONV2_%03d"%(i+1)

            if os.path.exists ("time_series/%s/eyetracking_ts/%s_%s.pkl"%(subject, testBlock, conv)) and os.path.exists ("time_series/%s/eyes_gradient_ts/%s_%s.pkl"%(subject, testBlock, conv)):
                continue

            """ the gradient of the eye mouvement """
            fx = np.array (convers [i]) [:,1:3]. astype (float)
            x  = np.array (convers [i]) [:,0]. astype (float)
            gradient = np. gradient (fx, x, axis = 0)
            gradient = np. concatenate ((np. reshape (x, (-1, 1)), gradient), axis = 1)
            gradient = to_df (resample_ts (gradient, index, mode = "mean"))
            gradient. columns = ["Time (s)", "Vx", "Vy"]


            """ Resampling data according to the videos frequency """
            long_data = resample_ts (np.array (convers [i]) [:,0:3]. astype (float), long_index, mode = "mean")


            """ long df for eye tracking synchronization """
            long_df = pd.DataFrame (long_data, columns = colnames)

            #print (long_df)


            #for col in long_df.columns:
                #df[col]. fillna (df[col].mean(), inplace=True)

            # Filter the  signal
            '''s = long_df [["x"]]. values
            t = long_df ["Time (s)"]. values
            print (s)
            filtered = denoise_signal (s, level = 1)

            # Plot results
            fig, ax = plt.subplots (2,1,figsize=(10,6))
            ax[0].plot(t, s, 'r', label = 'original')
            ax[1].plot(t, filtered, 'g', label = 'filtered')

            plt.xlabel("Time [s]")

            fig. legend ()
            plt.tight_layout()
            plt.show()

            exit (1)'''


            #df. to_pickle ("time_series/%s/eye_ts/%s_%s.pkl"%(subject, testBlock, conv))
            long_df. to_pickle ("time_series/%s/eyetracking_ts/%s_%s.pkl"%(subject, testBlock, conv))
            gradient. to_pickle ("time_series/%s/eyes_gradient_ts/%s_%s.pkl"%(subject, testBlock, conv))
