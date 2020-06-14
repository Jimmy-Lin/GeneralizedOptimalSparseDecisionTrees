import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from model.gosdt import GOSDT
from model.pygosdt import PyGOSDT
from model.osdt import OSDT
from model.visualizer import plot2DClassifier, plot2DConfidence

dataframe = pd.DataFrame(pd.read_csv('experiments/preprocessed/compas-binary.csv'))

n = 10

X = pd.DataFrame(dataframe.iloc[:,:-1])
y = pd.DataFrame(dataframe.iloc[:,-1])

hyperparameters = {
    # "objective": "acc",
    # "precision": 2,
    "profile_output": 'experiments/profiles/profile.csv',
    "regularization": 0.005,
    "uncertainty_tolerance": 0.0,
    "workers": 1,
    "output_limit": 1,
    "time_limit": 300,
    "verbose": True,
    "optimism": 0.7,
    "equity": 0.5,
    "sample_depth": 1,
    "opencl_platform_index": -1,
    "opencl_device_index": -1
}

model = GOSDT(hyperparameters, preprocessor="none")
start = time.time()
model.fit(X, y)
finish = time.time()
print("Execution Time: {}".format(finish - start))

prediction = model.predict(X)
training_accuracy = model.score(X, y)
print("Training Accuracy: {}".format(training_accuracy))
print(model.tree)

# plot2DClassifier('GOSDT with TreeEncoder', X, y, model)