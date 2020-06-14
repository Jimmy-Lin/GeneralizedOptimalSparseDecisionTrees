import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from math import ceil, floor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

from model.encoder import Encoder
from model.gosdt import GOSDT
from model.cart import CART

def impute(X):
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(X)
    return imp.transform(X)

def random_forest_feature_selection(dataset, k, balance=False):
    dataframe = pd.DataFrame(pd.read_csv(dataset, delimiter=",")).dropna()
    X = pd.get_dummies(dataframe[dataframe.columns[0:-1]], prefix_sep='=?=')
    y = dataframe[dataframe.columns[-1]]
     # This needs to be deterministic or it will risk our reproducibility
    if balance:
        clf = RandomForestClassifier(n_estimators=10, max_features=None, class_weight="balanced_subsample", random_state=0)
    else:
        clf = RandomForestClassifier(n_estimators=10, max_features=None, random_state=0)

    clf.fit(X, y)  
    encoded_importances = clf.feature_importances_
    decoded_importances = [0 for j in range(dataframe.shape[1]-1)]
    for i, importance in enumerate(clf.feature_importances_):
        encoded_title = X.dtypes.index[i]
        decoded_title = encoded_title if encoded_title in dataframe.columns else encoded_title.split("=?=")[0]
        decoded_importances[list(dataframe.columns).index(decoded_title)] += importance
    indices = np.argsort(decoded_importances)[::-1]
    print(decoded_importances)
    print(indices)
    selected_indices = [indices[i] for i in range(k)]
    # selected_indices = sorted(indices[i] for i in range(k))
    print("{} out of {} features selected for {} dataset".format(len(selected_indices), len(dataframe.columns), dataset))
    return list(dataframe.columns[selected_indices])

# def decision_tree_tuner(X, y, balance=None):
#     k = 10
#     initial = 0.1
#     reduction = 0.5
#     regularizations = [0.10, 0.5, 0.01, 0.005]
#     X = pd.get_dummies(X, prefix_sep='=?=')
#     (n,m) = X.shape

#     selected = None

#     for regularization in regularizations:
#         kfolds = KFold(n_splits=k)

#         configuration = {
#             "max_depth": floor(1 / regularization),
#             "min_samples_split": max(2, ceil(regularization * 2 * n)),
#             "min_samples_leaf": ceil(regularization * n),
#             "max_leaf_nodes": max(2, floor(1 / (2 * regularization))),
#             "min_impurity_decrease": regularization,
#             "class_weight": balance
#         }

#         # Estimate test accuracy by X-validation
#         test_accuracies = []
#         for train_index, test_index in kfolds.split(X):
#             X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
#             X_test, y_test = X.iloc[test_index, :], y.iloc[test_index]

#             model = DecisionTreeClassifier(**configuration)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             if balance == "balanced":
#                 test_accuracies.append(balanced_accuracy_score(y_test, y_pred))
#             else:
#                 test_accuracies.append(accuracy_score(y_test, y_pred))

#         # Retrain on full data set and find the useful features
#         model = DecisionTreeClassifier(**configuration)
#         model.fit(X, y)


#         features = []        
#         for column in X.columns[list_features(model.tree_)].tolist():
#             feature = column.split("=?=")[0] 
#             if not feature not in features:
#                 features.append(features)

#         results = {
#             "regularization": regularization,
#             "median_test_accuracy": 0.5 * (sorted(test_accuracies)[4] + sorted(test_accuracies)[5]),
#             "features": features
#         }

#         if selected is None or selected["median_test_accuracy"] < results["median_test_accuracy"]:
#             selected = results
#     return selected


def decision_tree_tuner(dataset, l, balance=False):
    dataframe = shuffle(pd.DataFrame(pd.read_csv("experiments/datasets/{}/train.csv".format(dataset), delimiter=",")).dropna(), random_state=0)

    if l > 0:
        selected_features = random_forest_feature_selection(dataset, l, balance=balance)
    else:
        selected_features = dataframe.columns[0:-1].to_list()

    X = dataframe[selected_features]
    y = dataframe[dataframe.columns[-1]]
    
    k = 10
    regularizations = [0.5, 0.2, 0.1,
                0.09, 0.08,  0.07,  0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
                0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001
            ]
    selected = None

    for regularization in regularizations:
        kfolds = KFold(n_splits=k, random_state=0)

        # Estimate test accuracy by X-validation
        training_accuracies = []
        test_accuracies = []
        for train_index, test_index in kfolds.split(X):
            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index, :], y.iloc[test_index]

            model = GOSDT({"regularization": regularization})
            model.fit(X_train, y_train)
            training_accuracies.append(accuracy_score(y_train, model.predict(X_train)))
            test_accuracies.append(accuracy_score(y_test, model.predict(X_test)))

        # Retrain on full data set and find the useful features
        model = GOSDT({"regularization": regularization})
        model.fit(X.iloc[:, :], y.iloc[:])
        results = {
            "regularization": regularization,
            "median_test_accuracy": sorted(test_accuracies)[2],
            "median_training_accuracy": sorted(training_accuracies)[2],
            "features": selected_features
        }
        print(results)

        if selected is None or selected["median_test_accuracy"] < results["median_test_accuracy"]:
            selected = results
    
    return selected

# Code to automatically select features and regularization levels

# Parses the DecisionTreeClassifier from Sci-Kit Learn according to their documentation
# Returns the number of leaves in this model
# Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
def list_features(tree):
    selected = []
    for f in tree.feature:
        if not abs(f) in selected:
            selected.append(abs(f))
    return selected
