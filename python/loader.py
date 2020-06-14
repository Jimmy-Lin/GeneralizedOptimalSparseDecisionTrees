import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.optimal_sparse_decision_forest import OptimalSparseDecisionForest

dataframe = pd.DataFrame(pd.read_csv('experiments/datasets/iris/original.csv', delimiter=","))
# dataframe = pd.DataFrame(pd.read_csv('experiments/datasets/compas_preprocessed/original.csv', delimiter=","))
# Select the first two features to make training easier
X = dataframe[dataframe.columns[0:-1]]
# Select the last column which is a categorical feature of 3 distinct classes
y = dataframe[dataframe.columns[-1:]]

forest = OptimalSparseDecisionForest()
forest.load("model.json")

print("Number of Models Produced: {}".format(len(forest.trees)))

for i, tree in enumerate(forest.trees):
    prediction = tree.predict(X)
    training_accuracy = 1 - tree.error(X, y)
    print("Training Accuracy (Excluding Complexity Penalty): {}".format(training_accuracy))
    # plot2DClassifier("Tree {}".format(i), X, y, tree)
    if i > 10:
        print("Too many models produced, truncating visualization")
        break
