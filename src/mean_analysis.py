import os
import glob
import pandas as pd
import numpy as np

from mat4py import loadmat

#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
    files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))
    return sorted (files)

#----------------------------------------------------------------#
def generate_stats (subject, type, colnames):
    files = process_transcriptions (subject, type)

    if not os. path. exists ("stats_ts/%s"%subject):
    	os. makedirs ("stats_ts/%s"%subject)

    HH_data = []
    HR_data = []

    for filename in files:
        data = pd. read_pickle (filename)

        if data. shape [0] < 50:
            print ("Subject: %s, the conversation %s have less than 50 lines"%(subject, filename))
        if "CONV1" in filename:
            HH_data. append (data. mean (axis = 0). loc [colnames]. tolist ())
        if "CONV2" in filename:
            HR_data. append (data. mean (axis = 0). loc [colnames]. tolist ())

    if type == "speech_ts":
        type = "speech_right_ts"

    elif type == "speech_left_ts":
        colnames = [a. split ('_')[0] for a in colnames]

    HH_data = pd.DataFrame (HH_data, columns = colnames)
    HR_data = pd.DataFrame (HR_data, columns = colnames)

    '''HH_data. to_csv ("stats_ts/%s/%s_HH.csv"%(subject, type), sep = ';', index = False)
    HR_data. to_csv ("stats_ts/%s/%s_HR.csv"%(subject, type), sep = ';', index = False)'''

    return HH_data, HR_data


#=============================================
if __name__ == '__main__':

    #=======================================================
    #   define the subjects to process: the participants
    #=======================================================
    subjects = ["sub-%02d"%i for i in range (1, 26)]
    subjects. remove ("sub-01")
    subjects. remove ("sub-19")
    subjects. remove ("sub-25")
    subjects. remove ("sub-04")

    #==========================
    #    Process fMRI data
    #==========================
    fmri_data = loadmat("data/hypothalamus_physsocial.mat")
    fmri_hh = []
    fmri_hr = []
    for i in range (len (subjects)):
        fmri_hh. extend (np. array (fmri_data ["hypothal"][i]) [:,[0,2,4]]. flatten ())
        fmri_hr. extend (np. array (fmri_data ["hypothal"][i]) [:,[1,3,5]]. flatten ())


    #==========================
    #    Process speech data
    #==========================
    colnames = ["IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters", "LexicalRichness1", "LexicalRichness2", "Polarity", "Subjectivity"]
    colnames_left = [a + "_left" for a in colnames]

    if not os. path. exists ("stats_ts"):
    	os. makedirs ("stats_ts")

    speech_hh_data = pd. DataFrame (columns = colnames)
    speech_hr_data = pd. DataFrame (columns = colnames)

    for subject in subjects:
        subj_hh_data, subj_hr_data = generate_stats (subject, "speech_ts", colnames)
        speech_hh_data = pd. concat ([speech_hh_data, subj_hh_data], axis = 0, ignore_index = True)
        speech_hr_data = pd. concat ([speech_hr_data, subj_hr_data], axis = 0, ignore_index = True)

    #=====================================
    #   Concatenate data and save them
    #=====================================

    fmri_speech_hh_data = pd. concat ([pd. DataFrame (fmri_hh, columns = ["hypothal"]), speech_hh_data], axis = 1)
    fmri_speech_hr_data = pd. concat ([pd. DataFrame (fmri_hr, columns = ["hypothal"]), speech_hr_data], axis = 1)

    print (fmri_speech_hh_data. shape)
    print (fmri_speech_hh_data. columns)
