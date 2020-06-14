import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from model.gosdt import GOSDT

def optimize(model, name, columns, workers):

    dataframe = shuffle(pd.DataFrame(pd.read_csv("experiments/datasets/{}/train.csv".format(name), delimiter=",")))
    X = dataframe[columns]
    y = dataframe[dataframe.columns[-1:]]

    with open("experiments/configurations/default.json", 'r') as default_source:
        defaults = default_source.read()
    hyperparameters = json.loads(defaults)
    hyperparameters.update({ "workers": workers })
    result = open("experiments/analysis/{}_data/regularization/{}.csv".format(model, name), "w")
    result.write("regularization,leaves,training_accuracy,test_accuracy\n")

    k = 10
    initial = 0.1
    reduction = 0.5
    regularizations = [ initial * pow(reduction, i) for i in range(k) ]
    kfolds = KFold(n_splits=k)
    index = 0

    max_test_accuracy = 0.0
    maximizing_regularization = 1.0

    # What if we "flip" the train-test split?
    # Smaller training set allows us to train faster at the cost of accuracy
    # However, the enlarged test set means the test accuracy is less noise and better reflects the true accuracy
    for train_index, test_index in kfolds.split(X):
        regularization = regularizations[index]
        X_train, y_train = X.iloc[train_index, :], y.iloc[train_index, :]
        X_test, y_test = X.iloc[test_index, :], y.iloc[test_index, :]
        hyperparameters["regularization"] = regularization

        model = GOSDT(hyperparameters)
        model.fit(X_train, y_train, subprocessing=True)
        model = model.trees[0] # Select the first tree

        training_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        leaves = len(model)
        result.write("{},{},{},{}\n".format(regularization, leaves, training_accuracy, test_accuracy))
        print("Regularization: {}, Leaves: {}, Training Accuracy: {}, Test Accuracy: {}".format(regularization, leaves, training_accuracy, test_accuracy))

        if test_accuracy >= max_test_accuracy:
            max_test_accuracy = test_accuracy
            maximizing_regularization = regularization

        index += 1

    result.close()
    return maximizing_regularization

def load(model, name):
    dataset = pd.DataFrame(pd.read_csv('experiments/analysis/{}_data/regularization/{}.csv'.format(model, name), delimiter=","))
    dataset.sort_values(by=['test_accuracy'], ascending=False, inplace=True)
    return dataset['regularization'].iloc[0]