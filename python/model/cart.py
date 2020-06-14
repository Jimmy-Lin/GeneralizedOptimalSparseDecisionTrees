import numpy as np
import pandas as pd
import time
from math import ceil, floor
from sklearn.tree import DecisionTreeClassifier
from model.tree_classifier import TreeClassifier
from model.encoder import Encoder


class CART:
    def __init__(self, preprocessor="complete", regularization=None, depth=1, support=1, width=1):
        # Precompute serialized configuration
        self.preprocessor = "complete"
        self.regularization = regularization
        self.depth = depth
        self.support = support
        self.width = width

    def fit(self, X, y):
        (n, m) = X.shape

        encoder = Encoder(X.values[:,:], header=X.columns[:], mode=self.preprocessor, target=y[y.columns[0]])
        headers = encoder.headers
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)
        self.encoder = encoder

        if not self.regularization is None:
            regularization = self.regularization
            config = {
                "max_depth": floor(1 / regularization) if regularization > 0 else None,
                "min_samples_split": max(ceil(regularization * 2 * n), 2),
                "min_samples_leaf": max(1, ceil(regularization * n)),
                "max_leaf_nodes": floor(1 / (2 * regularization)) if regularization > 0 else None,
                "min_impurity_decrease": regularization
            }
            model = DecisionTreeClassifier(**config)
        else:
            config = {
                "max_depth": self.depth,
                "min_samples_leaf": self.support,
                "max_leaf_nodes": self.width
            }
            model = DecisionTreeClassifier(**config)

        start = time.perf_counter()
        model.fit(X, y)
        self.duration = time.perf_counter() - start
        source = self.__translate__(model.tree_)
        self.tree = TreeClassifier(source, encoder=encoder, X=X, y=y)
        return self
    
    def __translate__(self, tree, id=0, depth=-1):
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        if tree.children_left[id] != children_right[id]:
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
                "prediction": 0 if len(tree.value[id][0]) == 1 or tree.value[id][0][0] >= tree.value[id][0][1] else 1
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