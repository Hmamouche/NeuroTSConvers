import os
import glob
import pandas as pd

#=============================================
# generate time series from transcriptions files
def process_transcriptions (subject, type = "speech_ts"):
    print ("\t" + subject, 15*'-', '\n')

    files = glob. glob ("time_series/%s/%s/*.pkl"%(subject, type))

    for i in range (len (files)):
        if "_CONV2" in files [i]:
            files [i] = files [i]. replace ("_CONV2", "")
        elif "_CONV1" in files [i]:
            files [i] = files [i]. replace ("_CONV1", "")

    return files

def generate_stats (subject, type, colnames):
    conversations = sorted (process_transcriptions (subject, type))
    print (conversations)
    exit (1)
    if not os. path. exists ("stats_ts/%s"%subject):
    	os. makedirs ("stats_ts/%s"%subject)
    out_data = []
    for conv in conversations:

        if int (conv[-7:-4]) % 2 == 1:
            filename = conv[0:-7] + "CONV1_" + conv[-7:-4] + ".pkl"
        elif int (conv[-7:-4]) % 2 == 0:
            filename = conv[0:-7] + "CONV2_" + conv[-7:-4] + ".pkl"
        #print (filename)
        data = pd. read_pickle (filename)
        out_data. append (data. mean (axis = 0).loc [colnames]. tolist ())


    out_data = pd.DataFrame (out_data, columns = colnames)
    out_data. to_csv ("stats_ts/%s/%s.csv"%(subject, type), sep = ';', index = False)
#=============================================
if __name__ == '__main__':

    subjects = []
    for i in range (1, 26):
    	if i < 10:
    		subjects. append ("sub-0%s"%str(i))
    	else:
    		subjects. append ("sub-%s"%str(i))

    colnames = ["Signal", "IPU", "Overlap", "ReactionTime", "FilledBreaks", "Feedbacks", "Discourses", "Particles", "Laughters", "LexicalRichness1", "LexicalRichness2"]
    regions = ["region_%d"%i for i in range (1,278)]

    if not os. path. exists ("stats_ts"):
    	os. makedirs ("stats_ts")

    for subject in subjects:
        #generate_stats (subject, "speech_left_ts", colnames)
        generate_stats (subject, "physio_ts", regions)
