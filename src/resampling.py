import numpy as np

#==========================================
def regroupe_data (list_, mode):
    """
        reducing a list of lists into one list by means of
        mean, mode, max, or binary
    """
    for row in list_:
    	if mode == "mean":
    	    return np.nanmean (list_, axis=0). tolist ()
    	elif mode == "max":
    	    return np.nanmax (list_, axis=0). tolist ()
    	elif mode == "mode":
    	    return sc_mode (list_, axis=0, nan_policy = 'omit')[0][0]. tolist ()
    	elif mode == "binary":
    		res = np.nanmean (list_, axis=0). tolist ()
    		for i in range (len (res)):
    			if res [i] > 0:
    				res [i] = 1
    			else:
    				res [i] = 0
    		return res

#=============================================================
def resample_ts (data, index, mode = "mean"):

    """
        Resampling a time series according to an index.
        data : a list of lists (observations), or a 2D np array, representing the input time series.
                the first column must contain the index of the data.
        the data must contain an index in the first column
        index : the new index (with smaller frequency compared to index of data)
        return  an resampled numpy array
    """

    resampled_ts = []
    rows_ts = []
    j = 0

    for i in range (len (data)):
    	if j >= len (index):
    	    break

    	if (data[i][0] > index [j]):
    		if len (rows_ts) == 0:
    			resampled_ts. append ([index [j]] + [0 for i in range (len (data [0][1:]) )])
    		else:
    			resampled_ts. append ([index [j]] + regroupe_data (rows_ts, mode))
    		initializer = 0
    		j += 1
    		rows_ts = []

    	rows_ts. append (data [i][1:])

    if len (rows_ts) > 0 and j < len (index):
        resampled_ts. append ([index [j]] + regroupe_data (rows_ts, mode))

    return np. array (resampled_ts)
