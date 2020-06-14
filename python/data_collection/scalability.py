import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from math import ceil
from sklearn.model_selection import KFold
from time import time

from model.gosdt import GOSDT

def scalability(model, name, columns, max_workers, regularization):
    dataframe = pd.DataFrame(pd.read_csv("experiments/datasets/{}/train.csv".format(name), delimiter=","))
    timing_output = "experiments/analysis/{}_data/scalability/{}.csv".format(model, name)
    result = open(timing_output, "w")
    # Intentionally omit the newline so that later lines can make up for it
    result.write("dimensions,workers,time\n")
    result.flush()
    result.close()

    max_time = 1800

    with open("experiments/configurations/default.json", "r") as default_source:
        defaults = default_source.read()
    hyperparameters = json.loads(defaults)
    hyperparameters = {
        "time_limit": max_time,
        "output_limit": 0,
        "regularization": regularization,
        "timing_output": timing_output
    }

    max_dimensions = len(columns)
    for j in range(1, max_dimensions + 1):
        timeout = False
        workers = max_workers
        while workers >= 1:
            k = workers
            X = dataframe[columns[0:j]]
            y = dataframe[dataframe.columns[-1:]]
            hyperparameter = hyperparameters["workers"] = k
            if timeout == True:
                result = open(timing_output, "a")
                result.write("{},{},{}\n".format(j, k, max_time))
                result.flush()
                result.close()
                print("Dimensions = {}, Workers = {}, Timeout".format(j, k))
                continue
            try:
                result = open(timing_output, "a")
                result.write("{},{},".format(j, k))
                result.flush()
                result.close()
                model = GOSDT(hyperparameters)
                model.fit(X, y, subprocessing=True)  # This call must insert the timing
            except Exception as e:
                print(e)
                result = open(timing_output, "a")
                result.write("{}\n".format(max_time))
                result.flush()
                result.close()
                print("Dimensions = {}, Workers = {}, Timeout".format(j, k))
                timeout = True
                # break
            else:
                result = open(timing_output, "a")
                result.write("\n")
                result.flush()
                result.close()
                print("Dimensions = {}, Workers = {}".format(j, k))

            if workers == 1:
                break
            else:
                workers = ceil(workers / 2.0)
    result.close()
