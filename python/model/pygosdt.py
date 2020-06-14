import json
import pandas as pd
import time
from subprocess import Popen, PIPE
from numpy import array
from sklearn.metrics import confusion_matrix, accuracy_score
from os import remove

# local imports
from .pygosdt_lib.models.parallel_osdt_classifier import ParallelOSDTClassifier
from .encoder import Encoder
from .tree_classifier import TreeClassifier

class PyGOSDT:
    def __init__(self, configuration={}, preprocessor="complete"):
        self.configuration = configuration
        self.configuration["output_limit"] = 1
        self.preprocessor = preprocessor
        self.encoder = None
        if not "objective" in self.configuration:
            self.configuration["objective"] = "accuracy"

    def load(self, path):
        with open(path, 'r') as model_source:
            result = model_source.read()
        result = json.loads(result)
        self.trees = [ TreeClassifier(source, encoder=self.encoder) for source in result ]
        self.tree = self.trees[0]

    def train(self, X, y):
        start = time.perf_counter()
                
        hyperparameters = {
            # Regularization coefficient which effects the penalty on model complexity
            'regularization': self.configuration["regularization"],

            'max_depth': float('Inf'),  # User-specified limit on the model
            'max_time': self.configuration["time_limit"],  # User-specified limit on the runtime

            'workers': 1,  # Parameter that varies based on how much computational resource is available

            'visualize_model': False,  # Toggle whether a rule-list visualization is rendered
            'visualize_training': False,  # Toggle whether a dependency graph is streamed at runtime
            'verbose': False,  # Toggle whether event messages are printed
            'log': False,  # Toggle whether client processes log to logs/work_<id>.log files
            'profile': False,  # Toggle Snapshots for Profiling Memory Usage

            'configuration': {  # More configurations around toggling optimizations and prioritization options

                # 'objective': 'balanced_accuracy', # Choose from accuracy, balanced_accuracy, weighted_accuracy
                # 'accuracy_weight': 0.7, # Only used for weighted accuracy

                'priority_metric': 'depth',  # Decides how tasks are prioritized
                # Decides how much to push back a task if it has pending dependencies
                'deprioritization': 0.01,

                'warm_start': False, # Warm start with cart tree's risk as upperbound

                # Note that Leaf Permutation Bound (Theorem 6) is
                # Toggles the assumption about objective independence when composing subtrees (Theorem 1)
                # Disabling this actually breaks convergence due to information loss
                'hierarchical_lowerbound': True,
                # Toggles whether problems are pruned based on insufficient accuracy (compared to other results) (Lemma 2)
                'look_ahead': True,
                # Toggles whether a split is avoided based on insufficient support (proxy for accuracy gain) (Theorem 3)
                'support_lowerbound': True,
                # Toggles whether a split is avoided based on insufficient potential accuracy gain (Theorem 4)
                'incremental_accuracy_lowerbound': True,
                # Toggles whether a problem is pruned based on insufficient accuracy (in general) (Theorem 5)
                'accuracy_lowerbound': True,
                # Toggles whether problem equivalence is based solely on the capture set (Similar to Corollary 6)
                'capture_equivalence': True,
                # Hamming distance used to propagate bounding information of similar problems (Theorem 7 + some more...)
                "similarity_threshold": 0,
                # Toggles whether equivalent points contribute to the lowerbound (Proposition 8 and Theorem 9)
                'equivalent_point_lowerbound': True,

                # Toggles compression of dataset based on equivalent point aggregation
                'equivalent_point_compression': True,
                # Toggles whether asynchronous tasks can be cancelled after being issued
                'task_cancellation': False,
                # Toggles whether look_ahead prunes using objective upperbounds (This builds on top of look_ahead)
                'interval_look_ahead': True,
                # Cooldown timer (seconds) on synchornization operations
                'synchronization_cooldown': float('Inf'),
                # Probability of saying "Fine. I will do it myself."
                'independence': 1.0
            }
        }

        if "profile_output" in self.configuration:
            hyperparameters["configuration"]["profile_output"] = self.configuration["profile_output"]

        model = ParallelOSDTClassifier(**hyperparameters)
        self.duration = model.fit(X, y)

        self.tree = TreeClassifier(model.source(), encoder=self.encoder)

    def fit(self, X, y, subprocess=True):
        X = X.copy()
        y = y.copy()

        encoder = Encoder(X.values[:,:], header=X.columns[:], mode=self.preprocessor, target=y[y.columns[0]])
        headers = encoder.headers
        X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
        y = y.reset_index(drop=True)

        self.encoder = encoder

        self.train(X.values[:,:], y.values[:,-1])
        return self

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

