import sys
import glob
import os

from joblib import Parallel, delayed
import multiprocessing
import argparse

#====================================#

def usage():
    print ("execute the script with -h for usage.")

#====================================#
# generate time series from transcriptions files
def process_transcriptions (subject, left):
    print ("\t" + subject, 15*'-', '\n')

    if left:
        out_dir = "time_series/" + subject + "/speech_left_ts/"
    else:
        out_dir = "time_series/" + subject + "/speech_ts/"

    if not os. path. exists (out_dir):
        os. makedirs (out_dir)

    conversations = glob. glob ("data/transcriptions/" + subject + "/*")
    conversations. sort ()

    for conv in conversations:
        try:
            if left:
                os. system ("python3 src/generate_ts/speech_features.py %s %s --left" % (conv, out_dir))
            else:
                os. system ("python3 src/generate_ts/speech_features.py %s %s" % (conv, out_dir))
        except:
            print ("Error in processing %s"%conv)

    print (subject + "Done ....")

#====================================#
# generate time series from videos

def process_videos (subject, type = "e"):

    print ("\t" + subject, 15*'-', '\n')
    out_dir_emotions = "time_series/" + subject + "/emotions_ts/"

    out_dir_landMarks = "time_series/" + subject + "/facial_features_ts/"
    out_dir_eyetracking = "time_series/" + subject + "/eyetracking_ts/"
    out_dir_colors = "time_series/" + subject + "/colors_ts/"

    for out_dir in [out_dir_emotions, out_dir_landMarks, out_dir_eyetracking, out_dir_colors]:
    	if not os. path. exists (out_dir):
    		os. makedirs (out_dir)

    videos = glob. glob ("data/videos/" + subject + "/*.avi")
    videos. sort ()

    for video in videos:
    	try:
    		if type == "eye":
    			os. system ("python3 src/generate_ts/eyetracking.py " + video + " " + out_dir_eyetracking)

    		elif type == 'e':
    			os.system("python3 src/generate_ts/generate_emotions_ts.py " +  video + " " + out_dir_emotions)

    		elif type == 'f':
    			os. system ("python3 src/generate_ts/facial_action_units.py " +  video + " " + out_dir_landMarks)

    		if type == "c":
    			os. system ("python3 src/generate_ts/colorfulness.py " + video + " " + out_dir_colors)
    	except:
    		print ("Error in processing video%s"%video)

#====================================#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument("subject", help="the subject name (for example sub-01), 'all' (default) to process all the subjects.", default="all")
    parser. add_argument ('--subjects', '-s', nargs = '+', type=int)
    parser.add_argument("--transcriptions","-t",  help="Process transcriptions.", action="store_true")
    parser.add_argument("--emotions", "-e", help="Process emotions.", action="store_true")
    parser.add_argument("--colors", "-c", help="Images colors.", action="store_true")
    parser.add_argument("--eyetracking", "-eye", help="Process eye tracking.", action="store_true")
    parser.add_argument("--facial", "-f", help="Process landmarks.", action="store_true")
    parser.add_argument("--left", "-le", help="Process participant speech.", action="store_true")

    args = parser.parse_args()

    if args.subjects == [0]:
        subjects = ["sub-%02d"%i for i in range (2, 24)]
    else:
        subjects = ["sub-%02d"%i for i in args.subjects]

    if not os. path. exists ("time_series"):
    	os. makedirs ("time_series")


    nax_cores = multiprocessing.cpu_count() - 1

    #try:
    if args. transcriptions:
    	Parallel (n_jobs = 5) (delayed(process_transcriptions) (subject, args.left) for subject in subjects)

    if args. colors:
    	Parallel (n_jobs=6) (delayed(process_videos) (subject, 'c') for subject in subjects)

    if args. emotions:
    	Parallel (n_jobs=1) (delayed(process_videos) (subject, 'e') for subject in subjects)
    if args. facial:
    	Parallel (n_jobs=3) (delayed(process_videos) (subject, 'f') for subject in subjects)
    if args. eyetracking:
    	Parallel (n_jobs=2) (delayed(process_videos) (subject, 'eye') for subject in subjects)
    #except:
    	#print ("Error in Parallel loop")
