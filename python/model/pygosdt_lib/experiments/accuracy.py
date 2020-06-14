import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from model.pygosdt_lib.experiments.logger import Logger
from model.pygosdt_lib.data_structures.dataset import read_dataframe


def accuracy_analysis(dataset, model_class, hyperparameters, path):
    X = dataset.values[:, :-1]
    y = dataset.values[:, -1]

    # Perform cross validation over k-folds, one for each proposed hyperparameter
    if len(hyperparameters) == 1:
        hyperparameters = [hyperparameters[0] for _i in range(2)]
    kfolds = KFold(n_splits=len(hyperparameters))

    logger = Logger(path=path, header=['hyperparameter', 'width', 'accuracy'])

    model_index = 0
    for train_index, test_index in kfolds.split(X):
        X = dataset.values[:, :-1]
        y = dataset.values[:, -1]

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        hyperparameter = hyperparameters[model_index]

        model = model_class(**hyperparameter)

        try:
            model.fit(X_train, y_train)
        except:
            pass
        else:
            accuracy = model.score(X_test, y_test)

            # Compute Tree Width of the model
            if model_class == DecisionTreeClassifier:
                width = compute_width(model)
            else:
                width = model.width

            logger.log([str(hyperparameter), width, accuracy])
        model_index += 1


def plot_accuracy_analysis(dataset_name, title):
    fig = plt.figure(figsize=(10, 8), dpi=100)

    dataset = read_dataframe(
        'data/accuracy/{}/{}.csv'.format(dataset_name, 'cart'))
    (n, m) = dataset.shape
    accuracies = {}
    for i in range(n):
        width = dataset.values[i, 1]
        accuracy = dataset.values[i, 2]
        if not width in accuracies:
            accuracies[width] = accuracy
        else:
            accuracies[width] = max(accuracies[width], accuracy)
    x = list(sorted(accuracies.keys()))
    y = [accuracies[width] for width in x]
    plt.plot(x, y, label='cart', markersize=5, marker='o', linewidth=0)

    dataset = read_dataframe(
        'data/accuracy/{}/{}.csv'.format(dataset_name, 'osdt'))
    (n, m) = dataset.shape
    accuracies = {}
    for i in range(n):
        width = dataset.values[i, 1]
        accuracy = dataset.values[i, 2]
        if not width in accuracies:
            accuracies[width] = accuracy
        else:
            accuracies[width] = max(accuracies[width], accuracy)
    x = list(sorted(accuracies.keys()))
    y = [accuracies[width] for width in x]
    plt.plot(x, y, label='osdt', markersize=5, marker='o', linewidth=0)

    dataset = read_dataframe(
        'data/accuracy/{}/{}.csv'.format(dataset_name, 'parallel_osdt'))
    (n, m) = dataset.shape
    accuracies = {}
    for i in range(n):
        width = dataset.values[i, 1]
        accuracy = dataset.values[i, 2]
        if not width in accuracies:
            accuracies[width] = accuracy
        else:
            accuracies[width] = max(accuracies[width], accuracy)
    x = list(sorted(accuracies.keys()))
    y = [accuracies[width] for width in x]
    plt.plot(x, y, label='parallel_osdt',
             markersize=5, marker='o', linewidth=0)

    plt.xlabel('Tree Width')
    plt.ylabel('Test Accuracy')
    plt.grid()
    plt.legend()
    plt.title(title)

# Parses the DecisionTreeClassifier from Sci-Kit Learn according to their documentation
# Returns the number of leaves in this model
# Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
def compute_width(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    leaf_count = 0
    for i in range(n_nodes):
        if is_leaves[i]:
            leaf_count += 1
    return leaf_count
