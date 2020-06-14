import numpy as np
import pandas as pd
import time
from math import ceil, floor, log
from sklearn.metrics import confusion_matrix, accuracy_score
from dl85 import DL85Classifier
from model.tree_classifier import TreeClassifier
from model.encoder import Encoder


class DL85:
    def __init__(self, preprocessor="complete", regularization=None, depth=1, support=1, time_limit=900):
        # Precompute serialized configuration
        self.preprocessor = preprocessor
        self.regularization = regularization
        self.depth = depth
        self.support = support
        self.time_limit = time_limit

    def fit(self, X, y):
        self.shape = X.shape
        (n, m) = self.shape

        encoder = Encoder(X.values[:,:], header=X.columns[:], mode=self.preprocessor, target=y[y.columns[0]])
        headers = encoder.headers
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)
        self.encoder = encoder

        if not self.regularization is None:
            depth = ceil(1 / self.regularization) if self.regularization > 0 else 2**30
            support = ceil(self.regularization * n)
            
            def error(sup_iter):
                supports = list(sup_iter)
                maxindex = np.argmax(supports)
                return sum(supports) - supports[maxindex] + self.regularization * n, maxindex

            clf = DL85Classifier(
                fast_error_function=error,
                iterative=True, 
                time_limit=self.time_limit,
                min_sup=support,
                max_depth=depth
            )
        else:
            clf = DL85Classifier(
                iterative=True, 
                time_limit=self.time_limit,
                min_sup=self.support,
                max_depth=self.depth
            )

        start = time.perf_counter()
        clf.fit(X, y)
        self.duration = time.perf_counter() - start
        self.space = clf.lattice_size_

        source = self.__translate__(clf.tree_)
        self.tree = TreeClassifier(source, encoder=encoder, X=X, y=y)
        return self

    def __translate__(self, node):
        (n, m) = self.shape

        if "class" in node:
            return {
                "name": "class",
                "prediction": node["class"],
                "loss": node["error"] / n,
                "complexity": self.regularization
            }
        elif "feat" in node:
            return { 
                "feature": node["feat"], 
                "name": self.encoder.headers[node["feat"]],
                "relation": "==",
                "reference": 1,
                "true": self.__translate__(node["left"]),
                "false": self.__translate__(node["right"])
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