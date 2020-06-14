
# Overview: Module containing functions for hyerparameter optimization

# Summary: Train and select the best model over a list of different hyperparameter settings using cross validation
# Input:
#    dataset: Pandas dataframe containing n-1 columns of features followed by 1 column of labels
#    model_class: Python class implementing standard sci-kit-learn model interface as follows
#       __init__(self, hyperparameters)
#       fit(self, X_train, y_test)
#       score(X_test, y_test)
#   hyperparameters: list of dictionaries each containing keyword arguments holding hyperparameter assignments for model construction
#   retrain: flag to retrain on the full dataset using the optimal hyperparameters
# Output:
#   model: the model that scored the highest in test accuracy during cross-validation
#   accuracy: the test accuracy of the model that scored the highest
#   hyperparameter: the hyperparameter setting that resulted in the highest test accuracy

def train_cross_validate(dataset, model_class, hyperparameters=[{}], retrain=False):
    X = dataset.values[:, :-1]
    y = dataset.values[:, -1]

    # Perform cross validation over k-folds, one for each proposed hyperparameter
    if len(hyperparameters) == 1:
        hyperparameters = [hyperparameters[0] for _i in range(2)]
    kfolds = KFold(n_splits=len(hyperparameters))

    model_index = 0
    optimal_hyperparameter = None
    optimal_model = None
    optimal_accuracy = 0
    for train_index, test_index in kfolds.split(X):
        X = dataset.values[:, :-1]
        y = dataset.values[:, -1]

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        hyperparameter = hyperparameters[model_index]

        model = model_class(**hyperparameter)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        if accuracy >= optimal_accuracy:
            optimal_hyperparameters = hyperparameter
            optimal_model = model
            optimal_accuracy = accuracy

        model_index += 1

    # Note: This retrains the model over the full dataset, which breaks the association with the test accuracy
    if retrain == True:
        optimal_model = model_class(**optimal_hyperparameter)
        optimal_model.fit(X, y)

    return optimal_model, optimal_accuracy, optimal_hyperparameter
