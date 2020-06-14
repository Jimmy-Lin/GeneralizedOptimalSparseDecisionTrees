from data_collection.profile import profile
from data_collection.regularization import optimize, load
from data_collection.scalability import scalability
from data_collection.models import generate
from data_collection.feature_selection import (
    select_random_forest_features,
    select_decision_tree_features,
    select_all_features,
)
# Script used to collect data for the GOSDT model
# Hopefully we can produce similar automations around the other models

workers = 15
model = "gosdt"

def select_features(name, feature_count=None):
    if feature_count is None:
        return select_all_features(name)
    else:
        return select_random_forest_features(name, feature_count)

def tune(model, name, feature_count=None):
    columns = select_features(name, feature_count)
    print("Model: {}, Experiment: {}, Selected Features: {}".format(model, name, columns))
    print("Optimizing Regularization...")
    lamb = optimize(model, name, columns, workers)
    print("Optimal Regularization: {}".format(lamb))


def experiment(model, name, feature_count=None):
    columns = select_features(name, feature_count)
    print("Model: {}, Experiment: {}, Selected Features: {}".format(model, name, columns))
    print("Loading Optimal Regularization...")
    lamb = load(model, name) # Finds best regularizer from tuning
    print("Optimal Regularization: {}".format(lamb))
    generate(model, name, columns, workers, lamb) # generate models
    profile(model, name, columns, workers, lamb) # generate convergence profile
    scalability(model, name, columns, workers, lamb) # generate scalability results

# Easy experiments are ones where we can comfortably train on all features
easy_experiments = ["monk_1", "monk_2", "monk_3", "iris", "tic-tac-toe"]
easy_experiments = ["monk_1"]

# Hard experiments are ones where we haven't scaled up to all their features
# The additional numbers are current limits on number of features we can handle,
# followed by total number of features in the dataset
# (Most these are limited by memory capacity)
hard_experiments = [("coupon", 3, 26), ("compas", 2, 22), ("fico", 1, 23), ("adult", 1, 15)]

# Impossible experiments are ones where we cannot complete even a down-scaled run
    # nyclu: has some data type inconsistencies to investigate
    # broward: is so difficult that the RF feature selection is stalling :D
    # hcv: reached time limit with a single feature
    # netherlands_general: reaches memory with a single feature
    # netherlands_sexual: reaches memory with a single feature (BUT WE HIT +90% accuracy)
    # netherlands_violence: reaches memory with a single feature
impossible_experiments = [("nyclu", 1, 26), ("hcv", 1, 28), ("broward", 1, 23), 
    ("netherlands_general", 1, 11), ("netherlands_sexual", 1, 28), ("netherlands_violence", 1, 30)]

new_experiments = []

# High-Precision Datasets (Netherlands, HCV)

for name in easy_experiments:
    tune(model, name)
    experiment(model, name)

# for name, feature_count, _ in hard_experiments:
#     tune(model, name, feature_count=feature_count)
#     experiment(model, name, feature_count=feature_count)

# for name, feature_count, _ in new_experiments:
#     tune(model, name, feature_count=feature_count)