import numpy as np
import pandas as pd
import time
from math import ceil, floor, log
from sklearn.metrics import confusion_matrix, accuracy_score
from model.tree_classifier import TreeClassifier
from model.encoder import Encoder

import os
import json
import sys
import glob
from os import remove
from .rbinoct.learn_class_bin import main

class BinOCT:
    def __init__(self, preprocessor="complete", regularization=None, depth=1, time_limit=900):
        # Precompute serialized configuration
        self.preprocessor = preprocessor
        self.regularization = regularization
        self.depth = depth
        self.time_limit = time_limit

    def fit(self, X, y):

        encoder = Encoder(X.values[:,:], header=X.columns[:], mode=self.preprocessor, target=y[y.columns[0]])
        headers = encoder.headers
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)
        self.encoder = encoder

        timestamp = round(time.time())

        (n,m) = X.shape
        X.insert(m, "class", y)
        data_path = "binoct_{}.csv.tmp".format(timestamp)
        X.to_csv(data_path, index=False, sep=";")

        try:
            if not self.regularization is None:
                leaf_count = ceil(1 / self.regularization) if self.regularization > 0 else None
                depth = min(ceil(1 / self.regularization), ceil(log(leaf_count, 2))) if self.regularization > 0 else 2 ** 32
                start = time.perf_counter()
                main(["-f", data_path, "-d", depth, "-t", self.time_limit, "-z", self.regularization, "-n", n])
                self.duration = time.perf_counter() - start
            else:
                start = time.perf_counter()
                main(["-f", data_path, "-d", self.depth, "-t", self.time_limit, "-z", self.regularization, "-n", n])
                self.duration = time.perf_counter() - start

            with open(data_path + ".json") as result:
                source = json.load(result)
            self.tree = TreeClassifier(source, encoder=encoder, X=X, y=y)
        finally:
            remove(data_path)
            remove(data_path + ".json")
        return self
    
    def __translate__(self, tree, id=0, depth=-1):
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        if tree.children_left[id] != children_right[id]:
            print("Feature", tree.feature)
            return {
                "feature": abs(tree.feature[id]),
                "name": self.encoder.headers[abs(tree.feature[id])],
                "relation": "<=",
                "reference": tree.threshold[id],
                "true": self.__translate__(tree, id=tree.children_left[id], depth=depth+1),
                "false": self.__translate__(tree, id=tree.children_right[id], depth=depth+1)
            }
        else:
            return {
                "complexity": self.regularization,
                "loss": 0,
                "name": "class",
                "prediction": 0 if tree.value[id][0][0] >= tree.value[id][0][1] else 1
            }

    def predict(self, X):
        return self.tree.predict(X)

    def error(self, X, y, weight=None):
        return self.tree.error(X, y, weight=weight)

    def score(self, X, y, weight=None):
        return self.tree.score(X, y, weight=weight)

    def confusion(self, X, y, weight=None):
        return self.tree.confusion(self.predice(X), y, weight=weight)

    def latex(self):
        return self.tree.latex()

    def json(self):
        return self.tree.json()

    def binary_features(self):
        return len(self.encoder.headers)
        
    def __len__(self):
        return len(self.tree)

    def leaves(self):
        return self.tree.leaves()

    def nodes(self):
        return self.tree.nodes()

    def max_depth(self):
        return self.tree.maximum_depth()
        
    def regularization_upperbound(self, X, y):
        return self.tree.regularization_upperbound(X, y)

    def features(self):
        return self.tree.features()