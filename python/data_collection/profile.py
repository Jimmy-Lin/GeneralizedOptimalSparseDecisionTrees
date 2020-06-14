import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from model.gosdt import GOSDT

def profile(model, name, columns, workers, regularization):
    dataframe = pd.DataFrame(
        pd.read_csv("experiments/datasets/{}/train.csv".format(name), delimiter=",")
    )
    X = dataframe[columns]
    y = dataframe[dataframe.columns[-1:]]

    with open("experiments/configurations/default.json", 'r') as default_source:
        defaults = default_source.read()
    hyperparameters = json.loads(defaults)
    hyperparameters.update({
        "regularization": regularization,
        "output_limit": 0,
        "workers": workers,
        "profile_output": "experiments/analysis/{}_data/profile/{}.csv".format(model, name),
    })

    forest = GOSDT(hyperparameters)
    forest.fit(X, y, subprocessing=True)
