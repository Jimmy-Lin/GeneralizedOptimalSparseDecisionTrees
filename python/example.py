import pandas as pd
import numpy as np
from model.gosdt import GOSDT

dataframe = pd.DataFrame(pd.read_csv("experiments/datasets/iris/data.csv"))

X = dataframe[dataframe.columns[:-1]]
y = dataframe[dataframe.columns[-1:]]

hyperparameters = {
    "regularization": 0.04,
    "time_limit": 3600,
    "verbose": True
}

model = GOSDT(hyperparameters)
model.fit(X, y)
# model.load("python/model/model.json")
# model.load("../gosdt_icml/model.json")
print("Execution Time: {}".format(model.time))

prediction = model.predict(X)
training_accuracy = model.score(X, y)
print("Training Accuracy: {}".format(training_accuracy))
print("Size: {}".format(model.leaves()))
print("Loss: {}".format(1 - training_accuracy))
print("Risk: {}".format(
    model.leaves() * hyperparameters["regularization"]
    + 1 - training_accuracy))
model.tree.__initialize_training_loss__(X, y)
print(model.tree)
print(model.latex())