import os
import glob
import pandas as pd
import numpy as np
import argparse

from sklearn import preprocessing
from mat4py import loadmat
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
    files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))
    return sorted (files)

#===============================================
def get_unimodal_ts (subject, type):
    # do not consider the first columns (the time index)
    files = process_transcriptions (subject, type)

    HH_data = pd. DataFrame ()
    HR_data = pd. DataFrame ()

    for filename in files:
        data = pd. read_pickle (filename).iloc[:, 1:]

        if "CONV1" in filename:
            HH_data = HH_data. append (data, ignore_index = True)
        if "CONV2" in filename:
            HR_data = HR_data. append (data, ignore_index = True)

    index = [i for i in range (HH_data. shape [0])]
    return [HH_data. reindex (index), HR_data. reindex (index)]


#=============================================
if __name__ == '__main__':

    '''parser = argparse. ArgumentParser ()
    parser. add_argument ("--generate", "-g", help = "generate data", action="store_true")
    args = parser.parse_args()'''

    '''colnames_right = ["IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters", "Polarity", "Subjectivity"]
    colnames_left = [a + "_left" for a in colnames_right]
    colnames = colnames_left + colnames_right'''

    if not os.path.exists ("concat_time_series"):
        os.makedirs ("concat_time_series")

    subjects = ["sub-%02d"%i for i in range (1, 25)]
    #subjects = ["sub-11", "sub-13"]
    for sub in ["sub-01", "sub-19", "sub-04",  "sub-16"]:
        if sub in subjects:
            subjects. remove (sub)

    behavioral_hh_data = pd. DataFrame ()
    behavioral_hr_data = pd. DataFrame ()

    bold_hh_data = pd. DataFrame ()
    bold_hr_data = pd. DataFrame ()

    discr_bold_hh_data = pd. DataFrame ()
    discr_bold_hr_data = pd. DataFrame ()

    for subject in subjects:

        subj_bold = get_unimodal_ts (subject, "physio_ts")
        subj_discr_bold = get_unimodal_ts (subject, "discretized_physio_ts")
        subj_behavioral = get_unimodal_ts (subject,  "speech_left_ts")

        for type in ["speech_ts",  "eyetracking_ts"]:
            unimodal_data = get_unimodal_ts (subject, type)

            subj_behavioral[0] = pd. concat ([subj_behavioral [0] , unimodal_data [0]], axis = 1)
            subj_behavioral[1] = pd. concat ([subj_behavioral [1], unimodal_data [1]], axis = 1)

        behavioral_hh_data = behavioral_hh_data. append (subj_behavioral[0], ignore_index=True, sort=False)
        behavioral_hr_data = behavioral_hr_data. append (subj_behavioral[1], ignore_index=True, sort=False)

        bold_hh_data = bold_hh_data. append (subj_bold [0], ignore_index=True, sort=False)
        bold_hr_data = bold_hr_data. append (subj_bold [1], ignore_index=True, sort=False)

        discr_bold_hh_data = discr_bold_hh_data. append (subj_discr_bold [0], ignore_index=True, sort=False)
        discr_bold_hr_data = discr_bold_hr_data. append (subj_discr_bold [1], ignore_index=True, sort=False)

    print (behavioral_hh_data. shape)
    exit (1)
    behavioral_hh_data. to_csv ("concat_time_series/behavioral_hh_data.csv", sep = ';', index = False)
    behavioral_hr_data. to_csv ("concat_time_series/behavioral_hr_data.csv", sep = ';', index = False)

    bold_hh_data. to_csv ("concat_time_series/bold_hh_data.csv", sep = ';', index = False)
    bold_hh_data. to_csv ("concat_time_series/bold_hr_data.csv", sep = ';', index = False)

    discr_bold_hh_data. to_csv ("concat_time_series/discr_bold_hh_data.csv", sep = ';', index = False)
    discr_bold_hr_data. to_csv ("concat_time_series/discr_bold_hr_data.csv", sep = ';', index = False)
