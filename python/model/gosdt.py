import json
import pandas as pd
import time
from subprocess import Popen, PIPE
from numpy import array
from sklearn.metrics import confusion_matrix, accuracy_score
from os import remove
from .imbalance.osdt_imb_v9 import bbound, predict

from .encoder import Encoder
from .tree_classifier import TreeClassifier

class GOSDT:
    def __init__(self, configuration={}, preprocessor="complete"):
        self.configuration = configuration
        self.configuration["output_limit"] = 1
        self.preprocessor = preprocessor
        self.encoder = None

    def load(self, path):
        with open(path, 'r') as model_source:
            result = model_source.read()
        result = json.loads(result)
        self.trees = [ TreeClassifier(source, encoder=self.encoder) for source in result ]
        self.tree = self.trees[0]

    def native_train(self, X, y):
        from gosdt import fit

        # Serialize for extension
        (n,m) = X.shape
        dataset = X
        X.insert(m, "class", y)

        start = time.perf_counter()
        result = fit(dataset.to_csv(index=False), json.dumps(self.configuration, separators=(',', ':'))) # Perform extension call
        self.duration = time.perf_counter() - start

        result = json.loads(result) # Deserialize result
        # Parse result into model
        self.trees = [ TreeClassifier(source, encoder=self.encoder) for source in result ]
        self.tree = self.trees[0]

    def subprocess_train(self, X, y):
        timestamp = round(time.time())
        (n,m) = X.shape
        # X.insert(m, y.columns[0], list(y[y.columns[0]]))
        X.insert(m, "class", y)
        data_path = "data_{}.csv.tmp".format(timestamp)
        X.to_csv(data_path, index=False)
        temporary = not "output" in self.configuration
        if temporary:
            self.configuration["output"] = "models_{}.json.tmp".format(timestamp)
        config_path = "config_{}.json.tmp".format(timestamp)
        timing_path = "timing_{}.tmp".format(timestamp)
        self.configuration["timing_output"] = timing_path

        config = open(config_path, "w")
        config.write(json.dumps(self.configuration, separators=(',', ':')))
        config.flush()
        config.close()
        try:
            Popen(["gosdt", data_path, config_path]).wait()
            time.sleep(0.5)
            self.load(self.configuration["output"])
            t = open(self.configuration["timing_output"], "r")
            self.duration = float(t.read())
            t.close()
        finally:
            remove(data_path)
            remove(config_path)
            if temporary:
                remove(self.configuration["output"])

    def local_train(self, X, y):
        start = time.perf_counter()
        leaves_c, pred_c, dic_c, nleaves_c, m_c, n_c, totaltime_c, time_c, R_c, \
        COUNT_c, C_c, accu_c, best_is_cart_c, clf_c, \
        len_queue, time_queue, time_realize_best_tree, R_best_tree, count_tree= \
        bbound(X, y, 
                self.configuration["objective"], self.configuration["regularization"], 
                prior_metric='curiosity', 
                w=self.configuration["w"], theta=self.configuration["theta"], 
                MAXDEPTH=float('Inf'), MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
                support=True, incre_support=True, accu_support=False, equiv_points=True,
                lookahead=True, lenbound=True, R_c0 = 1, timelimit=self.configuration["time_limit"], init_cart = False,
                saveTree = False, readTree = False)
        self.duration = time.perf_counter() - start

        if best_is_cart_c:
            source = self.__translate_cart__(clf_c.tree_)
        else:
            decoded_leaves_c = []
            for leaf_c in leaves_c:
                decoded_leaf = tuple((dic_c[j] if j > 0 else -dic_c[-j]) for j in leaf_c)
                decoded_leaves_c.append(decoded_leaf)
            source = self.__translate__(dict(zip(decoded_leaves_c, pred_c)))
        self.tree = TreeClassifier(source, encoder=self.encoder)

    def fit(self, X, y, subprocess=True):
        X = X.copy()
        y = y.copy()

        encoder = Encoder(X.values[:,:], header=X.columns[:], mode=self.preprocessor, target=y[y.columns[0]])
        headers = encoder.headers
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)

        self.encoder = encoder

        if "objective" in self.configuration and self.configuration["objective"] in {"acc", "bacc", "wacc", "f1", "auc_convex", "partial_auc"}:
            if self.configuration["objective"] == "acc":
                self.configuration["theta"] = None
                self.configuration["w"] = None
            elif self.configuration["objective"] == "bacc":
                self.configuration["theta"] = None
                self.configuration["w"] = None
            elif self.configuration["objective"] == "wacc":
                self.configuration["theta"] = None
            elif self.configuration["objective"] == "f1":
                self.configuration["theta"] = None
                self.configuration["w"] = None
            elif self.configuration["objective"] == "auc":
                self.configuration["theta"] = None
                self.configuration["w"] = None
            elif self.configuration["objective"] == "pauc":
                self.configuration["w"] = None
            # print("Specialized Objective Selected. Using OSDT Python Implementation")
            print("Sequential")
            self.local_train(X.values[:,:], y.values[:,-1])
        else:
            if subprocess == False:
                self.native_train(X, y)
            else:
                self.subprocess_train(X, y)
        return self

    def __translate__(self, leaves):
        if len(leaves) == 1:
            return {
                "complexity": self.configuration["regularization"],
                "loss": 0,
                "name": "class",
                "prediction": list(leaves.values())[0]
            }
        else:
            features = {}
            for leaf in leaves.keys():
                if not leaf in features:
                    for e in leaf:
                        features[abs(e)] = 1
                    else:
                        features[abs(e)] += 1
            split = None
            max_freq = 0
            for feature, frequency in features.items():
                if frequency > max_freq:
                    max_freq = frequency
                    split = feature
            positive_leaves = {}
            negative_leaves = {}
            for leaf, prediction in leaves.items():
                if split in leaf:
                    positive_leaves[tuple(s for s in leaf if s != split)] = prediction
                else:
                    negative_leaves[tuple(s for s in leaf if s != -split)] = prediction
            return {
                "feature": split,
                "name": self.encoder.headers[split],
                "reference": 1,
                "relation": "==",
                "true": self.__translate__(positive_leaves),
                "false": self.__translate__(negative_leaves),
            }

    def __translate_cart__(self, tree, id=0, depth=-1):
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
                "true": self.__translate_cart__(tree, id=tree.children_left[id], depth=depth+1),
                "false": self.__translate_cart__(tree, id=tree.children_right[id], depth=depth+1)
            }
        else:
            return {
                "complexity": self.configuration["regularization"],
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
        return self.tree.confusion(self.predict(X), y, weight=weight)

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

