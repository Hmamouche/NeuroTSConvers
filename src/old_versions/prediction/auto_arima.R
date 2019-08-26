# Author: Youssef Hmamouche

# Use compiler and jit for this file (when processing multiple files this improves performance)
require(compiler)
enableJIT(3)

## Prediction using the auto.arima model

library(forecast)
library(stringr)
require("reticulate")

source_python ("src/utils/read_pickle.py")

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

#####
args = commandArgs(trailingOnly=TRUE)
print (args)

if (length (args) < 4)
{
	print ("Unsuficiant args!")
	stop ()
}

### data
filename_1 = paste0 ("time_series/sub-", str_pad(args[1], 2, pad = "0"), "/physio_ts/convers-TestBlocks",args[2],"_CONV",args[3],"_", str_pad(args[4], 3, pad = "0"),".pkl")

filename_2 = paste0 ("time_series/sub-", str_pad(args[1], 2, pad = "0"), "/physio_ts/convers-TestBlocks",args[2],"_CONV2_002.pkl")

### Arima model
conv1 = read_pickle_file (filename_1)
conv2 = read_pickle_file (filename_2)
y1 = conv1 [,"region_73"]
y2 = conv2 [,"region_73"]
region_74 = conv1 [,"region_74"]

print ("-----------------------------")
model_1 = Arima (y = y1, order = c(1, 1, 1))
print (model_1)
print ("-----------------------------")
model_2 = Arima (y = y2, order = c(1, 1, 1), model=model_1)
print (model_2)
print ("-----------------------------")
model_2 = Arima (y = y2, order = c(1, 1, 1))
print (model_2)





