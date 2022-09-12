import json
import pandas as pd
import time
from numpy import array
from sklearn.metrics import confusion_matrix, accuracy_score

import gosdt.libgosdt as gosdt # Import the GOSDT extension
from gosdt.model.encoder import Encoder
from gosdt.model.imbalance.osdt_imb_v9 import bbound, predict # Import the special objective implementation
from gosdt.model.tree_classifier import TreeClassifier # Import the tree classification model

class GOSDT:
    def __init__(self, configuration={}):
        self.configuration = configuration
        self.time = 0.0
        self.stime = 0.0
        self.utime = 0.0
        self.maxmem = 0
        self.numswap = 0
        self.numctxtswitch = 0
        self.iterations = 0
        self.size = 0
        self.tree = None
        self.encoder = None
        self.lb = 0
        self.ub = 0
        self.timeout = False
        self.reported_loss = 0

    def load(self, path):
        """
        Parameters
        ---
        path : string
            path to a JSON file representing a model
        """
        with open(path, 'r') as model_source:
            result = model_source.read()
        result = json.loads(result)
        self.tree = TreeClassifier(result[0])

    def __train__(self, X, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains a model using the GOSDT native extension
        """
        (n, m) = X.shape
        dataset = X.copy()
        dataset.insert(m, "class", y) # It is expected that the last column is the label column

        gosdt.configure(json.dumps(self.configuration, separators=(',', ':')))
        result = gosdt.fit(dataset.to_csv(index=False)) # Perform extension call to train the model

        self.time = gosdt.time() # Record the training time
        self.stime = gosdt.stime()
        self.utime = gosdt.utime()

        if gosdt.status() == 0:
            print("gosdt reported successful execution")
            self.timeout = False
        elif gosdt.status() == 2:
            print("gosdt reported possible timeout.")
            self.timeout = True
            self.time = -1
            self.stime = -1
            self.utime = -1
        else :
            print('----------------------------------------------')
            print(result)
            print('----------------------------------------------')
            raise Exception("Error: GOSDT encountered an error while training")

        result = json.loads(result) # Deserialize resu

        self.tree = TreeClassifier(result[0]) # Parse the first result into model
        self.iterations = gosdt.iterations() # Record the number of iterations
        self.size = gosdt.size() # Record the graph size required

        self.maxmem = gosdt.maxmem()
        self.numswap = gosdt.numswap()
        self.numctxtswitch = gosdt.numctxtswitch()

        self.lb = gosdt.lower_bound() # Record reported global lower bound of algorithm
        self.ub = gosdt.upper_bound() # Record reported global upper bound of algorithm
        self.reported_loss = gosdt.model_loss() # Record reported training loss of returned tree

        print("training completed. {:.3f}/{:.3f}/{:.3f} (user, system, wall), mem={} MB".format(self.utime, self.stime, self.time, self.maxmem >> 10))
        print("bounds: [{:.6f}..{:.6f}] ({:.6f}) loss={:.6f}, iterations={}".format(self.lb, self.ub,self.ub - self.lb, self.reported_loss, self.iterations))

    def fit(self, X, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains the model so that this model instance is ready for prediction
        """
        if "objective" in self.configuration:
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
            else:
                raise Exception("Error: GOSDT does not support this accuracy objective")
            self.__python_train__(X, y)
        else:
            self.__train__(X, y)
        return self

    def predict(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the prediction associated with each row
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.predict(X)

    def error(self, X, y, weight=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
        weight : real number
            an n-by-1 column of weights to apply to each sample's misclassification
        Returns
        ---
        real number : the inaccuracy produced by applying this model overthe given dataset, with optionals for weighted inaccuracy
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.error(X, y, weight=weight)

    def score(self, X, y, weight=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
        weight : real number
            an n-by-1 column of weights to apply to each sample's misclassification
        Returns
        ---
        real number : the accuracy produced by applying this model overthe given dataset, with optionals for weighted accuracy
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.score(X, y, weight=weight)

    def confusion(self, X, y, weight=None):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
        weight : real number
            an n-by-1 column of weights to apply to each sample's misclassification
        Returns
        ---
        matrix-like, shape = [k_classes by k_classes] : the confusion matrix of all classes present in the dataset
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.confusion(self.predict(X), y, weight=weight)

    def __len__(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return len(self.tree)

    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.leaves()

    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.nodes()

    def max_depth(self):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree will return 1.
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.maximum_depth()

    def latex(self):
        """
        Note
        ---
        This method doesn't work well for label headers that contain underscores due to underscore being a reserved character in LaTeX
        Returns
        ---
        string : A LaTeX string representing the model
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.latex()

    def json(self):
        """
        Returns
        ---
        string : A JSON string representing the model
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.json()

    def __python_train__(self, X, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains a model using the GOSDT pure Python implementation modified from OSDT
        """

        encoder = Encoder(X.values[:,:], header=X.columns[:], mode="complete", target=y[y.columns[0]])
        headers = encoder.headers

        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)

        # Translation of Variables:
        # leaves_c := data representation of leaves using decision paths
        # pred_c := data representation of predictions
        # dic := leaf translator
        # nleaves := number of leaves
        # m := number of encoded features
        # n := number of samples
        # totaltime := total optimization run time (includes certification)
        # time_c := time-to-optimality
        # R_c := minimized risk
        # COUNT := number of models evaluated
        # C_c := number of models evaluated at optimality
        # accu := accuracy
        # best_is_cart := whether the optimal model is produced by cart
        # clf := prediction model produced by cart

        start = time.perf_counter()
        leaves_c, pred_c, dic, nleaves, m, n, totaltime, time_c, R_c, COUNT, C_c, accu, best_is_cart, clf = bbound(
            X.values[:,:], y.values[:,-1],
            self.configuration["objective"], self.configuration["regularization"],
            prior_metric='curiosity',
            w=self.configuration["w"], theta=self.configuration["theta"],
            MAXDEPTH=float('Inf'), MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
            support=True, incre_support=True, accu_support=False, equiv_points=True,
            lookahead=True, lenbound=True, R_c0 = 1, timelimit=self.configuration["time_limit"], init_cart = False,
            saveTree = False, readTree = False)

        self.duration = time.perf_counter() - start

        if best_is_cart:
            source = self.__translate_cart__(clf.tree_)
        else:
            decoded_leaves = []
            for leaf in leaves_c:
                decoded_leaf = tuple((dic[j] if j > 0 else -dic[-j]) for j in leaf)
                decoded_leaves.append(decoded_leaf)
            source = self.__translate__(dict(zip(decoded_leaves, pred_c)))
        self.tree = TreeClassifier(source, encoder=encoder)
        self.tree.__initialize_training_loss__(X, y)

    def __translate__(self, leaves):
        """
        Converts the leaves of OSDT into a TreeClassifier-compatible object
        """
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
                "name": "feature_" + str(split),
                "reference": 1,
                "relation": "==",
                "true": self.__translate__(positive_leaves),
                "false": self.__translate__(negative_leaves),
            }

    def __translate_cart__(self, tree, id=0, depth=-1):
        """
        Converts the CART results into a TreeClassifier-compatible object
        """
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        if tree.children_left[id] != children_right[id]:
            return {
                "feature": abs(tree.feature[id]),
                "name": "feature_" + str(split),
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
