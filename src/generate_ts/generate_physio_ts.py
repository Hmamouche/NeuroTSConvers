import scipy.io as sio
import pandas as pd
import numpy as np
from glob import glob
import os
import sys
import argparse


#======================================================

def nearestPoint(vect, value):
    dist = abs(value - vect[0])
    pos = 0

    for i in range(1, len(vect)):
        if abs(value - vect[i]) < dist:
            dist = abs(value - vect[i])
            pos = i

    return pos

#======================================================

def add_duration(df):
    duration = []

    for i in range(df.shape[0]):
        if i == (df.shape[0] - 1):
            duration.append([df.iloc[i, 3] / 1000.0, df.iloc[i, 3] / 1000.0, 0])
        else:
            duration.append([df.iloc[i, 3] / 1000.0, df.iloc[i + 1, 3] / 1000.0,
                             df.iloc[i + 1, 3] / 1000.0 - df.iloc[i, 3] / 1000.0])

    df = pd.concat([df, pd.DataFrame(duration, columns=['begin', 'fin', 'Interval'])], axis=1)

    return df

#======================================================

def convers_to_df (data, colnames, index, begin, end, type_conv, num_conv):
    index_normalized = index[begin:end]
    start_pt = index_normalized [0] # - 1.205 / 2

    #print (begin, end)
    for j in range(0, end - begin):
       index_normalized[j] -= start_pt

    convers_data = pd.DataFrame (data [begin:end]) #, index = index_normalized)

    #gradient = np. gradient (convers_data. values, axis = 0)
    diff = convers_data. diff (axis = 0, periods = 1)
    diff. iloc [0, :] = convers_data. iloc [0, :]

    rolling_mean = convers_data. rolling (window = 3). mean()
    rolling_mean. iloc [0: 2, :] = convers_data. iloc [0: 2,:]



    #convers_data. insert (0, "Real time", index[begin:end])
    filename = "time_series/" + subject + "/new_physio_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"
    diff_filename = "time_series/" + subject + "/physio_diff_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"
    smooth_filename = "time_series/" + subject + "/physio_smooth_ts/convers-" + testBlock + "_" + type_conv + "_" + "%03d"%num_conv + ".pkl"
    #convers_data = convers_data. diff ()


    #convers_data. set_index (index_normalized, inplace = True)
    convers_data. reset_index (inplace = True)
    convers_data.columns = ['Time (s)'] +  colnames

    diff. reset_index (inplace = True)
    diff. columns = ['Time (s)'] +  colnames

    rolling_mean. reset_index (inplace = True)
    rolling_mean. columns = ['Time (s)'] +  colnames

    #convers_data. fillna (0, inplace = True)

    convers_data.to_pickle (filename)
    diff.to_pickle (diff_filename)
    rolling_mean. to_pickle (smooth_filename)
    #convers_data.to_csv (filename, sep=';', index=False)
    #print (convers_data)

#======================================================
# --- USAGE

def usage():
    print ("execute the script with -h for usage.")

#======================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="the subject name (for example sub-01) or 'ALL' to process al the subjects.")

    args = parser.parse_args()

    _arg = args.subject

    subjects = []
    subjs = []

    if not os.path.exists("time_series"):
        os.makedirs("time_series")

    for i in range(1, 26):
        if i < 10:
            subjects.append("sub-0%s" % str(i))
        else:
            subjects.append("sub-%s" % str(i))

    if _arg == 'ALL' or _arg == 'all':
        subjs = subjects
        subjs.remove('sub-12')
        subjs.remove('sub-19')
        print (subjs)
    elif _arg in subjects:
        subjs = [_arg]
    else:
        usage()
        exit(1)

    for subject in subjs:

        for fname in ["new_physio_ts", "physio_diff_ts", "physio_smooth_ts"]:
            if not os.path.exists("time_series/%s/%s"%(subject, fname)):
                os.makedirs("time_series/%s/%s"%(subject, fname))

        print (subject, 15 * '*', '\n')

        num_subj = '0' + subject.split('-')[-1]

        #data_dat = sio.loadmat("data/physio_data/denoised/ROI_Subject" + num_subj + "_Condition000.mat")
        data_dat = sio.loadmat("data/new_physio_data/ROIdata/ROI_Subject" + num_subj + "_Condition000.mat")

        #print (data_dat['names']. tolist ())
        #exit (1)
        # ------------------ Analyse denoised data

        '''print ("the number of regions: %s"%str(data['data'][0].shape))
		print ("the number of voxels in each region: %s"%str(data['data'][0][0].shape[1]))
		print ("the length of voxels time series: %s"%str(data['data'][0][0].shape[0]))'''

        data = []

        for i in range(len(data_dat["data"][0][0][:, 0])):
            row = []
            for j in range(len(data_dat["data"][0])):
                row.append(data_dat["data"][0][j][i, 0])
            data.append(row)


        colnames = ["region_" + str(i + 1) for i in range (len(data[0]))]

        #index = [1.205 / 2.0]
        index = [0]
        for i in range (1, len (data)):
            index. append (1.205 + index [i - 1])

        # ------------------ Analyse log files

        log_files = glob("data/new_physio_data/logfiles/*" + subject + "*")
        testBlocks = ["" for i in range(4)]
        for i in range(4):
            for logfile in log_files:
                if "TestBlocks" + str(i + 1) in logfile:
                    testBlocks[i] = "TestBlocks" + str(i + 1)
                    break

        # if there is no logfile, we continue to the next subject
        if len(testBlocks) == 0:
            print ("subject %s has no logfiles" % subject)
            continue

        indice_block = 0
        for testBlock in testBlocks:
            log_block_file = glob ("data/new_physio_data/logfiles/*" + subject + "_task-convers-" + testBlock + "*") [0]
            df = pd.read_csv (log_block_file, sep='\t', header=0)
            df = df [['condition', 'image', 'duration', 'ONSETS_MS']]

            df = add_duration(df)


            hh_convers = df [df.condition.str.contains("CONV1")] [['condition', 'begin', 'fin']]
            hr_convers = df [df.condition.str.contains("CONV2")] [['condition', 'begin', 'fin']]

            nb_hh_convers = hh_convers. shape [0]
            nb_hr_convers = hr_convers. shape [0]

            hh = 1
            hr = 2

            for i in range(nb_hh_convers):
                begin = nearestPoint (index, hh_convers.values[i][1]) + (385 * indice_block)
                end = nearestPoint (index, hh_convers.values[i][2]) + (385 * indice_block)
                convers_to_df (data, colnames, index, begin, end, "CONV1", hh)
                hh += 2


            for i in range(nb_hr_convers):
                begin = nearestPoint (index, hr_convers.values[i][1]) + (385 * indice_block)
                end = nearestPoint (index, hr_convers.values[i][2]) + (385 * indice_block)
                convers_to_df (data, colnames, index, begin, end, "CONV2", hr)
                hr += 2
            indice_block += 1
