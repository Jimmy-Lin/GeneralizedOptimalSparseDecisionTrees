import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from model.gosdt import GOSDT

def generate(model, name, columns, workers, regularization):

    dataframe = shuffle(pd.DataFrame(pd.read_csv("experiments/datasets/{}/train.csv".format(name), delimiter=",")))
    X = dataframe[columns]
    y = dataframe[dataframe.columns[-1:]]

    with open("experiments/configurations/default.json", 'r') as default_source:
        defaults = default_source.read()
    hyperparameters = json.loads(defaults)
    hyperparameters.update({ "workers": workers })

    k = 10
    kfolds = KFold(n_splits=k)
    index = 0

    result = open("experiments/analysis/{}_data/regularization/{}.csv".format(model, name), "w")
    result.write("index,training_accuracy,test_accuracy\n")

    try:
        if not os.path.exists("experiments/analysis/{}/models/{}".format(model, name)):
            os.makedirs("experiments/analysis/{}_data/models/{}".format(model, name))
    except:
        pass

    for train_index, test_index in kfolds.split(X):
        X_train, y_train = X.iloc[train_index, :], y.iloc[train_index, :]
        X_test, y_test = X.iloc[test_index, :], y.iloc[test_index, :]
        hyperparameters["regularization"] = regularization

        clf = GOSDT(hyperparameters)
        clf.fit(X_train, y_train, subprocessing=True)
        clf = clf.trees[0] # Select the first tree

        training_accuracy = clf.score(X_train, y_train)
        test_accuracy = clf.score(X_test, y_test)
        result.write("{},{},{}\n".format(index, training_accuracy, test_accuracy))
        print("Training Accuracy: {}, Test Accuracy: {}".format(training_accuracy, test_accuracy))

        model_file = open("experiments/analysis/{}_data/models/{}/model_{}.pkl".format(model, name, index), "wb")
        pickle.dump(clf, model_file)
        model_file.close()

        index += 1    
    result.close()
    return