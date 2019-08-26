library (NlinTS)
library(stringr)
library (xtable)
require("reticulate")


# Normalisation
normalize <- function (F)
{
  for (i in 1 : ncol(F))
  {
    max = max(F[,i])
    min = min(F[,i])
    for(j in 1:nrow(F))
      F[j,i] = (F[j,i] - min) / (max - min)
  }
  return (F)
}
######################


args = commandArgs(trailingOnly=TRUE)
print (args)

if (length (args) < 4)
{
	print ("Unsuficiant args!")
	stop ()
}


### data

source_python ("read_pickle.py")

filename = paste0 ("time_series/sub-", str_pad(args[1], 2, pad = "0"), "/physio_ts/convers-TestBlocks",args[2],"_CONV",args[3],"_", str_pad(args[4], 3, pad = "0"),".pkl")
print (filename)

physio_ts = read_pickle_file (filename)
physio_ts = physio_ts[, 2:ncol (physio_ts)]

filename = paste0 ("time_series/sub-", str_pad(args[1], 2, pad = "0"), "/speech_ts/convers-TestBlocks",args[2],"_CONV",args[3],"_", str_pad(args[4], 3, pad = "0"),".pkl")
speech_ts = read_pickle_file (filename)

filename = paste0 ("time_series/sub-", str_pad(args[1], 2, pad="0"),"/emotions_ts/convers-TestBlocks",args[2],"_CONV",args[3],"_",str_pad(args[4],3, pad = "0"),".pkl")
emotions_ts = read_pickle_file (filename)


# Quantisze Emotions 
for (i in 1:ncol (emotions_ts))
{
	if (emotions_ts[i,3] == "None")
		emotions_ts[i,3] = 0
		
	else if (emotions_ts[i,3] == "neutral")
		emotions_ts[i,3] = 1

	else if (emotions_ts[i,3] == "surprise")
		emotions_ts[i,3] = 2
		
	else if (emotions_ts[i,3] == "sad")
		emotions_ts[i,3] = 3
		
	else if (emotions_ts[i,3] == "happy")
		emotions_ts[i,3] = 4
		
	else if (emotions_ts[i,3] == "fear")
		emotions_ts[i,3] = 5
		
	else if (emotions_ts[i,3] == "disgust")
		emotions_ts[i,3] = 6
		
	else if (emotions_ts[i,3] == "angry")
		emotions_ts[i,3] = 7
		
}

## Merge predictors
predictors = cbind (speech_ts[,2:ncol (speech_ts)], emotions_ts[,3])

predictors = normalize (predictors)
physio_ts = normalize (physio_ts)



caus_mat = data.frame (matrix (nrow = 10, ncol = ncol (predictors)))#, colnames (predictors))
colnames (caus_mat) = colnames (predictors)

te_mat = data.frame (matrix (nrow = 10, ncol = ncol (predictors)))#, colnames (predictors))
colnames (te_mat) = colnames (predictors)
colnames (te_mat)[ncol (te_mat)] = "emotions"
colnames (caus_mat)[ncol (te_mat)] = "emotions"


###	Transfer entropy
caus_vector = rep (0, ncol (predictors))
te_vector = rep (0, ncol (predictors))

rownames = rep ("", 0)
print ("Transfer entropy")

for (i in 1:10)
	rownames [i] = paste ("region_",i)
	
row.names (caus_mat) = rownames

for (i in 1:ncol (predictors))
{
	
}

print (te_vector)

### Granger causality
print ("Granger causality")
lag = 5
for (j in 1:10)
	{
	caus_vector = rep (0, ncol (predictors))
	te_vector = rep (0, ncol (predictors))
	for (i in 1:ncol (predictors))
	{	
	
		model = causality.test (physio_ts[,j], predictors[,i], lag = lag)
		
		#if (model$pvalue <= 0.1)
		caus_vector[i] = round (1 - model$pvalue, 3)
		#else
			#caus_vector[i] = 0
			
		te_vector[i] = round (te_cont (physio_ts[,j], predictors[,i], k= 3, p=5,q=5), 3)
			
		caus_mat [j,i] = caus_vector[i]
		te_mat [j,i] = te_vector[i]
	}
	}


print (xtable (caus_mat[,1:5]))
print (xtable (caus_mat[, 6:ncol (caus_mat)] ))
#print (xtable (te_mat))


## create output result
if (dir.exists("results/causalities") == 0)
	dir.create ("results/causalities")

output_dir = paste0 ("results/causalities/sub-", str_pad(args[1], 2, pad = "0"))
if (dir.exists (output_dir) == 0)
	dir.create (output_dir)

out_file = paste0 (output_dir, "/convers-TestBlocks",args[2],"_CONV",args[3],"_", str_pad (args[4], 3, pad = "0"), "_GC_lag=", lag, ".csv")
write.table(caus_mat, file = out_file,row.names=T,col.names=T, sep=";")

out_file = paste0 (output_dir, "/convers-TestBlocks",args[2],"_CONV",args[3],"_", str_pad (args[4], 3, pad = "0"), "_te_lag=", lag, ".csv")
write.table(te_mat, file = out_file,row.names=T,col.names=T, sep=";")


