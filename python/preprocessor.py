import sys
import os
import pandas as pd

from model.encoder import Encoder

# This script is used to convert regular CSV files into two separate (features and labels) binary CSV files.
# This preprocessing step is ONLY necessary when running the command-line build of the training program

# When the training using the Python interface, this preprocessing script is not necessary.
# Rhe encoder model is injected into the training process to hide the additional formatting from the user.

if not len(sys.argv) in {2, 4}:
    print("Usage python3 python/preprocessor.py [path to data set] ?[number of features] ?[number of labels]")
    exit()

path = sys.argv[1]
m = int(sys.argv[2])
w = int(sys.argv[3])
directory, basename = os.path.split(path)
dataframe = pd.DataFrame(pd.read_csv(path, delimiter=","))

if len(sys.argv) == 4:
    m = int(sys.argv[2])
    w = int(sys.argv[3])
elif len(sys.argv) == 2:
    m = dataframe.shape[1]-1
    w = 1

# encoder = Encoder(dataframe.values[:,:], header=dataframe.columns[:], mode="complete", target=dataframe.values[:,-1])
# encoded = pd.DataFrame(encoder.encode(dataframe.values[:,:]), columns=encoder.headers)
# encoded.to_csv(directory + '/complete.csv', index=False)

# encoder = Encoder(dataframe.values[:,:], header=dataframe.columns[:], mode="bucketize", target=dataframe.values[:,-1])
# encoded = pd.DataFrame(encoder.encode(dataframe.values[:,:]), columns=encoder.headers)
# encoded.to_csv(directory + '/bucketized.csv', index=False)

features = m
encoder = Encoder(dataframe.values[:,:features], header=dataframe.columns[:features], mode="radix")
encoded = pd.DataFrame(encoder.encode(dataframe.values[:,:features]), columns=encoder.headers)
encoded.insert(encoded.shape[1], "class", dataframe.values[:,-1])
encoded.to_csv(directory + '/radix.csv', index=False)

