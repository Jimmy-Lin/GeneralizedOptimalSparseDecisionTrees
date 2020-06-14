import numpy as np
import pandas as pd
import json
import time
import random
import sys
from math import ceil
from model.cart import CART
from model.dl85 import DL85
from model.binoct import BinOCT
from model.pygosdt import PyGOSDT
from model.gosdt import GOSDT
from model.osdt import OSDT
from model.encoder import Encoder
from data_collection.feature_selection import decision_tree_tuner
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from time import sleep

workers = 1
time_limit = 300

def tune(dataset, number_of_features):
    dataframe = shuffle(
        pd.DataFrame(
            pd.read_csv(
                "experiments/datasets/{}/train.csv".format(dataset), delimiter=","
            )
        ).dropna(),
        random_state=0,
    )
    encoder = Encoder(dataframe.values[:, -1], header=dataframe.columns[-1])

    X = dataframe[dataframe.columns[0:-1]]
    y = pd.DataFrame(encoder.encode(dataframe.values[:, -1]), columns=encoder.headers)[
        encoder.headers[0]
    ]

    result = decision_tree_tuner(dataset, number_of_features)
    print(
        "dataset: {}, balance: {}, test accuracy: {}, regularization: {}, selected features: {}".format(
            dataset,
            False,
            result["median_test_accuracy"],
            result["regularization"],
            result["features"],
        )
    )
    with open("experiments/datasets/{}/config.json".format(dataset), "w") as outfile:
        json.dump(result, outfile)

non_terminal = {
    "tic-tac-toe": 1,
    "coupon": 1,
    "compas": 1,
    "fico": 1,
    "netherlands_general": 1,
    #    "netherlands_sexual": 1,
    #    "netherlands_violence": 1
}

datasets = {
    # "monk_1": 0,
    "monk_2": 0,
    # "monk_3": 0,
    # "iris": 2,
    # "adult": 1,
}

methods = {
    # "cart": CART,
    # "dl85": DL85,
    # "binoct": BinOCT,
    # "gosdt": GOSDT,
}

def optimality_example(): # Section 3.2
    dataframe = pd.DataFrame(pd.read_csv("experiments/datasets/anon/train.csv")).dropna()
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])

    result = open("experiments/results/optimality_example.csv", "w")
    result.write("discretization,optimization,training_accuracy(%),nodes,leaves\n")

    model = BinOCT(depth=2, preprocessor="complete", time_limit=time_limit)
    model.fit(X, y)
    accuracy = model.score(X, y) * 100
    result.write("{},{},{},{},{}\n".format("midpoints","BinOCT",accuracy,model.nodes(),model.leaves()))

    model = BinOCT(depth=2, preprocessor="bucketize", time_limit=time_limit)
    model.fit(X, y)
    accuracy = model.score(X, y) * 100
    result.write("{},{},{},{},{}\n".format("bucketize","BinOCT",accuracy,model.nodes(),model.leaves()))

    model = DL85(depth=2, preprocessor="complete", time_limit=time_limit)
    model.fit(X, y)
    accuracy = model.score(X, y) * 100
    result.write("{},{},{},{},{}\n".format("midpoints","DL8.5",accuracy,model.nodes(),model.leaves()))

    model = DL85(depth=2, preprocessor="bucketize", time_limit=time_limit)
    model.fit(X, y)
    accuracy = model.score(X, y) * 100
    result.write("{},{},{},{},{}\n".format("bucketize","DL8.5",accuracy,model.nodes(),model.leaves()))

    model = GOSDT({ "regularization": 0.08, "time_limit": time_limit, "workers": workers }, preprocessor="complete")
    model.fit(X, y)
    accuracy = model.score(X, y) * 100
    result.write("{},{},{},{},{}\n".format("midpoints","GOSDT",accuracy,model.nodes(),model.leaves()))

    model = GOSDT({ "regularization": 0.08, "time_limit": time_limit, "workers": workers }, preprocessor="bucketize")
    model.fit(X, y)
    accuracy = model.score(X, y) * 100
    result.write("{},{},{},{},{}\n".format("bucketize","GOSDT",accuracy,model.nodes(),model.leaves()))

    result.close()

##########################
## ACCURACY vs SPARSITY ##
##########################

def trial(dataset, algorithm): # Section 5.1
    result = open("experiments/results/trial_{}_{}.csv".format(dataset, algorithm), "w")
    result.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

    dataframe = pd.DataFrame(pd.read_csv("experiments/preprocessed/{}.csv".format(dataset))).dropna()
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    (n, m) = X.shape

    configurations = {
        "size": [(1,2), (2,4), (3,8), (4,16), (5,32), (6,64)],
        "depth": [1, 2, 3, 4, 5, 6],
        "regularization": [ 0.2, 0.1,
            0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
            0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
            0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001
        ]
    }

    def record(generator):
        kfolds = KFold(n_splits=5, random_state=0)
        fold_index = 0
        for train_index, test_index in kfolds.split(X):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            (samples, features) = X_train.shape
            binary_features = len(Encoder(X_train.values[:,:], header=X_train.columns[:]).headers)
            try:
                model = generator()
                model.fit(X_train, y_train)
            except Exception as e:
                print(str(e))
                result.write("{},{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(fold_index,samples,features,binary_features))
            else:
                training_accuracy = model.score(X_train, y_train) * 100
                test_accuracy = model.score(X_test, y_test) * 100
                row = [
                    fold_index, samples, features, binary_features,
                    model.duration,
                    model.max_depth(), model.leaves(), model.nodes(),
                    training_accuracy, test_accuracy,
                    model.latex()
                ]
                print(*row)
                result.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(*row))
            fold_index += 1

    
    if algorithm == "cart":
        for depth, width in configurations["size"]:
            record(lambda : CART(depth=depth, width=width, preprocessor="none"))
            
    elif algorithm == "binoct":
        for depth in configurations["depth"]:
            record(lambda : BinOCT(depth=depth, time_limit=time_limit, preprocessor="none"))                
        
    elif algorithm == "dl85":
        for depth in configurations["depth"]:
            record(lambda : DL85(depth=depth, time_limit=time_limit, preprocessor="none"))                

    elif algorithm == "osdt":
        for regularization in configurations["regularization"]:
            record(lambda : OSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }, preprocessor="none"))                

    elif algorithm == "pygosdt":
        for regularization in configurations["regularization"]:
            record(lambda : PyGOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }, preprocessor="none"))                

    elif algorithm == "gosdt":
        for regularization in configurations["regularization"]:
            record(lambda : GOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }, preprocessor="none"))                

    result.close()

########################
## SAMPLE SCALABILITY ##
########################

def scale_samples(dataset, algorithm): # Section 5.2.1
    result = open("experiments/results/samples_{}_{}.csv".format(dataset, algorithm), "w")
    result.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

    dataframe = shuffle(pd.DataFrame(pd.read_csv("experiments/scalability/{}.csv".format(dataset))).dropna(), random_state=0)
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    (n, m) = X.shape

    configurations = {
        "samples": [
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
            45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
            100, 110, 120, 130, 140, 150, 160, 170,
            180, 200, 220, 240, 260, 280, 300,
            325, 350, 375, 400, 425, 450, 475, 500,
            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
            1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
            2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000,
            5000, 6000, 7000, 8000, 9000, 10000,
            11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
            22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000,
            45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000,
            100000, 110000, 1200000, 130000, 140000, 150000, 160000, 170000, 180000, 190000,
            200000, 220000, 2400000, 260000, 280000, 300000, 325000, 350000, 375000, 400000,
            500000, 600000, 700000, 800000, 10000000
        ],
        "size": (5,3),
        "depth": 5,
        "regularization": 0.03125
    }

    timeout = {}
    timeout_limit = 100

    def record(generator, train_index, test_index, method):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]
        (samples, features) = X_train.shape
        binary_features = len(Encoder(X_train.values[:,:], header=X_train.columns[:]).headers)
        if method in timeout and timeout[method] >= timeout_limit:
            result.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
            return
        try:
            model = generator()
            model.fit(X_train, y_train)
            if model.duration > time_limit:
                if method in timeout:
                    timeout[method] += 1
                else:
                    timeout[method] = 1 
        except Exception as e:
            print(str(e))
            result.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
            if method in timeout:
                timeout[method] += 1
            else: 
                timeout[method] = 1
        else:
            training_accuracy = model.score(X_train, y_train) * 100
            test_accuracy = model.score(X_test, y_test) * 100
            row = [
                samples, features, binary_features,
                model.duration,
                model.max_depth(), model.leaves(), model.nodes(),
                training_accuracy, test_accuracy,
                model.latex()
            ]
            result.write("NA,{},{},{},{},{},{},{},{},{},{}\n".format(*row))
            print(*row)    

    for sample_size in configurations["samples"]:
        if sample_size > n:
            break
        train_index = [ i for i in range(sample_size) ]
        test_index = [ i for i in range(sample_size, n) ]
        if algorithm == "cart":
            depth, width = configurations["size"]
            record(lambda : CART(depth=depth, width=width), train_index, test_index, "cart")
                
        elif algorithm == "binoct":
            depth = configurations["depth"]
            record(lambda : BinOCT(depth=depth, time_limit=time_limit), train_index, test_index, "binoct")                
            
        elif algorithm == "dl85":
            depth = configurations["depth"]
            record(lambda : DL85(depth=depth, time_limit=time_limit), train_index, test_index, "dl85")                

        elif algorithm == "osdt":
            regularization = configurations["regularization"]
            record(lambda : OSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }), train_index, test_index, "osdt")                

        elif algorithm == "pygosdt":
            regularization = configurations["regularization"]
            record(lambda : PyGOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }), train_index, test_index, "pygosdt")                

        elif algorithm == "gosdt":
            regularization = configurations["regularization"]
            record(lambda : GOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }), train_index, test_index, "gosdt")                

    result.close()


#########################
## FEATURE SCALABILITY ##
#########################

def scale_features(dataset, algorithm): # Section 5.2.1
    result = open("experiments/results/features_{}_{}.csv".format(dataset, algorithm), "w")
    result.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

    dataframe = shuffle(pd.DataFrame(pd.read_csv("experiments/scalability/{}.csv".format(dataset))).dropna(), random_state=0)
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    (n, m) = X.shape

    encoder = Encoder(X.values[:,:], header=X.columns[:])
    X = pd.DataFrame(encoder.encode(X.values[:,:]), columns=encoder.headers)
    (n, z) = X.shape

    sample_size = int( 0.9 * n )
    train_index = [ i for i in range(sample_size) ]
    test_index = [ i for i in range(sample_size, n) ]
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    configurations = {
        "features": [
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
            42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
            65, 70, 75, 80, 85, 90, 95, 100, 105, 110,
            120, 130, 140, 150, 160, 170, 180, 190,
            210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
            325, 350, 375, 400, 425, 450, 475, 500,
            525, 550, 575, 600, 625, 650, 675, 700,
            750, 800, 850, 900, 950, 1000,
            1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500,
            1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000,
            3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000,
            6000, 7000, 8000, 9000, 10000,
            11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
            21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000,
            32000, 34000, 36000, 38000, 40000, 42000, 44000, 46000, 48000, 50000,
            55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000
        ],
        "size": (5,32),
        "depth": 5,
        "regularization": 0.02
    }

    timeout = {}
    timeout_limit = 100

    def record(generator, feature_index, method):
        (samples, features) = X_train.shape
        binary_features = len(feature_index)
        if method in timeout and timeout[method] >= timeout_limit:
            result.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
            return
        try:
            model = generator()
            model.fit(X_train.iloc[:, feature_index], y_train)
            if model.duration > time_limit:
                if method in timeout:
                    timeout[method] += 1
                else:
                    timeout[method] = 1
        except Exception as e:
            print(str(e))
            result.write("NA,{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(samples,features,binary_features))
            if method in timeout:
                timeout[method] += 1
            else:
                timeout[method] = 1
        else:
            training_accuracy = model.score(X_train.iloc[:, feature_index], y_train) * 100
            test_accuracy = model.score(X_test.iloc[:, feature_index], y_test) * 100
            row = [
                samples, features, binary_features,
                model.duration,
                model.max_depth(), model.leaves(), model.nodes(),
                training_accuracy, test_accuracy,
                model.latex()
            ]
            result.write("NA,{},{},{},{},{},{},{},{},{},{}\n".format(*row))
            print(*row)    

    for k in configurations["features"]:
        if k > z:
            k = z

        feature_index = [ i for i in range(k) ]

        if algorithm == "cart":
            depth, width = configurations["size"]
            record(lambda : CART(depth=depth, width=width), feature_index, "cart")
                
        elif algorithm == "binoct":
            depth = configurations["depth"]
            record(lambda : BinOCT(depth=depth, time_limit=time_limit), feature_index, "binoct")                
            
        elif algorithm == "dl85":
            depth = configurations["depth"]
            record(lambda : DL85(depth=depth, time_limit=time_limit), feature_index, "dl85")                

        elif algorithm == "osdt":
            regularization = configurations["regularization"]
            record(lambda : OSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }), feature_index, "osdt")                

        elif algorithm == "pygosdt":
            regularization = configurations["regularization"]
            record(lambda : PyGOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }), feature_index, "pygosdt")

        elif algorithm == "gosdt":
            regularization = configurations["regularization"]
            record(lambda : GOSDT({ "regularization": regularization, "time_limit": time_limit, "workers": workers }), feature_index, "gosdt")                

        if k == z:
            break
    result.close()


def performance(dataset, algorithm): # Section 5.1
    result = open("experiments/results/performance_{}_{}.csv".format(dataset, algorithm), "w")
    result.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

    dataframe = pd.DataFrame(pd.read_csv("experiments/preprocessed/{}.csv".format(dataset))).dropna()
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    (n, m) = X.shape

    configurations = {
        "size": [(1,2), (2,4), (3,8), (4,16), (5,32), (6,64), (7,128)],
        "depth": [1, 2, 3, 4, 5, 6, 7],
        "regularization": [ 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128 ]
    }

    def record(generator):
        kfolds = KFold(n_splits=5, random_state=0)
        fold_index = 0
        for train_index, test_index in kfolds.split(X):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            (samples, features) = X_train.shape
            binary_features = len(Encoder(X_train.values[:,:], header=X_train.columns[:]).headers)
            try:
                model = generator()
                model.fit(X_train, y_train)
            except Exception as e:
                print(str(e))
                result.write("{},{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(fold_index,samples,features,binary_features))
            else:
                training_accuracy = model.score(X_train, y_train) * 100
                test_accuracy = model.score(X_test, y_test) * 100
                row = [
                    fold_index, samples, features, binary_features,
                    model.duration,
                    model.max_depth(), model.leaves(), model.nodes(),
                    training_accuracy, test_accuracy,
                    model.latex()
                ]
                print(*row)
                result.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(*row))
            fold_index += 1

    extended_time_limit = 300

    if algorithm == "cart":
        for depth, width in configurations["size"]:
            record(lambda : CART(depth=depth, width=width, preprocessor="none"))
            
    elif algorithm == "binoct":
        for depth in configurations["depth"]:
            record(lambda : BinOCT(depth=depth, time_limit=extended_time_limit, preprocessor="none"))                
        
    elif algorithm == "dl85":
        for depth in configurations["depth"]:
            record(lambda : DL85(depth=depth, time_limit=extended_time_limit, preprocessor="none"))                

    elif algorithm == "osdt":
        for regularization in configurations["regularization"]:
            record(lambda : OSDT({ "regularization": regularization, "time_limit": extended_time_limit, "workers": workers }, preprocessor="none"))                

    elif algorithm == "pygosdt":
        for regularization in configurations["regularization"]:
            record(lambda : PyGOSDT({ "regularization": regularization, "time_limit": extended_time_limit, "workers": workers }, preprocessor="none"))                

    elif algorithm == "gosdt":
        for regularization in configurations["regularization"]:
            record(lambda : GOSDT({ "regularization": regularization, "time_limit": extended_time_limit, "workers": workers }, preprocessor="none"))                

    result.close()

def profile(dataset, algorithm): # Section 5.1
    dataframe = shuffle(pd.DataFrame(pd.read_csv("experiments/preprocessed/{}.csv".format(dataset))).dropna(), random_state=0)
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    (n, m) = X.shape

    extended_time_limit = 300

    configurations = { "regularization": [0.005] }

    def record(generator):
        model = generator()
        model.fit(X, y)

    if algorithm == "osdt":
        for regularization in configurations["regularization"]:
            record(lambda : OSDT({
                "regularization": regularization,
                "time_limit": extended_time_limit,
                "workers": workers,
                "profile_output": "experiments/profiles/{}_{}.csv".format(dataset,algorithm)
            }, preprocessor="none"))                

    elif algorithm == "pygosdt":
        for regularization in configurations["regularization"]:
            record(lambda : PyGOSDT({
                "regularization": regularization,
                "time_limit": extended_time_limit,
                "workers": workers,
                "profile_output": "experiments/profiles/{}_{}.csv".format(dataset,algorithm)
            }, preprocessor="none"))                

    elif algorithm == "gosdt":
        for regularization in configurations["regularization"]:
            record(lambda : GOSDT({
                "regularization": regularization,
                "time_limit": extended_time_limit,
                "workers": workers,
                "profile_output": "experiments/profiles/{}_{}.csv".format(dataset,algorithm)
            }, preprocessor="none"))                

def imbalance(dataset, algorithm): # Section 5.1
    result = open("experiments/results/imbalance_{}_{}.csv".format(dataset, algorithm), "w")
    result.write("fold_index,samples,features,binary_features,time,depth,leaves,nodes,training,test,tex\n")

    dataframe = pd.DataFrame(pd.read_csv("experiments/preprocessed/{}.csv".format(dataset))).dropna()
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])
    (n, m) = X.shape

    configurations = {
        "regularization": [ 0.01 ]
    }

    def record(generator):
        kfolds = KFold(n_splits=5, random_state=0)
        fold_index = 0
        for train_index, test_index in kfolds.split(X):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]
            (samples, features) = X_train.shape
            binary_features = len(Encoder(X_train.values[:,:], header=X_train.columns[:]).headers)
            try:
                model = generator()
                model.fit(X_train, y_train)
            except Exception as e:
                print(str(e))
                result.write("{},{},{},{},-1,-1,-1,-1,-1,-1,NA\n".format(fold_index,samples,features,binary_features))
            else:
                training_accuracy = model.score(X_train, y_train) * 100
                test_accuracy = model.score(X_test, y_test) * 100
                row = [
                    fold_index, samples, features, binary_features,
                    model.duration,
                    model.max_depth(), model.leaves(), model.nodes(),
                    training_accuracy, test_accuracy,
                    model.latex()
                ]
                print(*row)
                result.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(*row))
            fold_index += 1

    print(dataset, algorithm)

    if algorithm == "gosdt-accuracy":
        for regularization in configurations["regularization"]:
            record(lambda : GOSDT({ "regularization": regularization, "time_limit": 1800, "workers": workers }, preprocessor="none"))                

    elif algorithm == "gosdt-balanced":
        for regularization in configurations["regularization"]:
            record(lambda : GOSDT({ "objective": "bacc", "regularization": regularization, "time_limit": 1800, "workers": workers }, preprocessor="none"))                

    elif algorithm == "gosdt-f1":
        for regularization in configurations["regularization"]:
            record(lambda : GOSDT({ "objective": "f1", "regularization": regularization, "time_limit": 1800, "workers": workers }, preprocessor="none"))

    elif algorithm == "gosdt-auc":
        for regularization in configurations["regularization"]:
            record(lambda : GOSDT({ "objective": "auc", "regularization": regularization, "time_limit": 1800, "workers": workers }, preprocessor="none"))

    result.close()

# path = "experiments/datasets/iris/train.csv"
# features = ['petal_width', 'petal_length', 'sepal_length', 'sepal_width']
# target = ['class==iris-virginica']

# path = "experiments/datasets/coupon/train.csv"
# features = ['coupon', 'map', 'time', 'passanger', 'occupation', 'expiration', 'Restaurant20To50', 'gender', 'destination', 'car', 'temperature', 'CoffeeHouse', 'income', 'weather', 'toCoupon_GEQ15min', 'age', 'toCoupon_GEQ25min', 'education', 'maritalStatus', 'direction_same', 'RestaurantLessThan20', 'Childrennumber', 'Bar', 'CarryAway', 'toCoupon_GEQ5min', 'direction_opp']
# target = ['Y']

# path = "experiments/datasets/adult/train.csv"
# features = ['marital-status', 'fnlwgt', 'age', 'education-num', 'capital-gain', 'occupation', 'hours-per-week', 'workclass', 'capital-loss', 'education', 'native-country', 'relationship', 'race', 'sex']
# target = ['income-bracket<=50k']

# path = "experiments/datasets/fico/train.csv"
# features = ['ExternalRiskEstimate', 'PercentTradesWBalance', 'MSinceOldestTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NetFractionRevolvingBurden', 'PercentInstallTrades', 'NumRevolvingTradesWBalance', 'MSinceMostRecentInqexcl7days', 'MSinceMostRecentTradeOpen', 'NumTotalTrades', 'MSinceMostRecentDelq', 'NetFractionInstallBurden', 'PercentTradesNeverDelq', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'NumTradesOpeninLast12M', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTrades60Ever2DerogPubRec', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NumTrades90Ever2DerogPubRec']
# target = ['PoorRiskPerformance']

# path = "experiments/datasets/compas/train.csv"
# features = ['p_age_first_offense', 'p_current_age', 'p_arrest', 'p_misdem_count_person', 'p_charge', 'p_charge_violent', 'p_felony_count_person', 'p_probation', 'p_felprop_violarrest', 'p_misdemassault_arrest', 'p_n_on_probation', 'p_felassault_arrest', 'p_prison', 'p_weapons_arrest', 'p_prison30', 'p_juv_fel_count', 'p_current_on_probation', 'p_prob_revoke', 'p_jail30', 'p_sex_arrest', 'p_murder_arrest', 'p_famviol_arrest']
# target = ['recid']

# path = "experiments/datasets/netherlands_general/train.csv"
# features = ['dichtheid', 'lftinsz1inclvtt', 'lft2', 'leeftijd', 'delcuziv', 'cgebland', 'lnvgalguz', 'sekse', 'dum21plus', 'dum1120']
# target = ['rec4']

# path = "experiments/datasets/netherlands_sexual/train.csv"
# features = ['leeftijd', 'lft2', 'lftinsz1inclvtt', 'dichtheid', 'vgboete', 'vgzeden', 'vgvermgng', 'vgvernoo', 'cgebland', 'vggev', 'vggeweld', 'zvernoo', 'zgeweld', 'vgtaak', 'lnvgalguz', 'vgvermg', 'vgtrans', 'zopium', 'vgverkeer', 'zoverig', 'vgopium', 'zvermgng', 'dum1120', 'zverkeer', 'vgoverig', 'dum21plus', 'zvermg']
# target = ['rec4']

# path = "experiments/datasets/netherlands_violence/train.csv"
# features = ['lftinsz1inclvtt', 'lft2', 'dichtheid', 'leeftijd', 'lnvgalguz', 'vggev', 'vggeweld', 'vgvermgng', 'vgvernoo', 'cgebland', 'vgboete', 'delcuziv', 'vgtrans', 'vgtaak', 'vgoverig', 'vgverkeer', 'vgvermg', 'zvernoo', 'vgopium', 'zoverig', 'vgzeden', 'dum1120', 'sekse', 'zvermgng', 'zverkeer', 'zopium', 'zzeden', 'dum21plus', 'zvermg']
# target = ['rec4']

# path = "experiments/datasets/hcv/train.csv"
# features = ['RNA 4', 'ALT 48', 'ALT 12', 'RBC', 'Plat', 'AST 1', 'ALT 36', 'ALT 1', 'RNA Base', 'ALT 24', 'ALT4', 'RNA 12', 'Age', 'WBC', 'ALT after 24 w', 'RNA EF', 'RNA EOT', 'BMI', 'HGB', 'Baseline histological Grading', 'Epigastric pain', 'Nausea/Vomting', 'Gender', 'Diarrhea', 'Fatigue & generalized bone ache', 'Fever', 'Headache', 'Jaundice']
# target = ['Baselinehistological staging']


command = sys.argv[1]
if command == "trial":
    trial(*(sys.argv[2:4]))

elif command == "samples":
    scale_samples(*(sys.argv[2:4]))

elif command == "features":
    scale_features(*(sys.argv[2:4]))

elif command == "performance":
    for data in ['car-evaluation', 'compas-binary', 'fico-binary', 'monk1-train', 'monk2-train', 'monk3-train', 'tic-tac-toe', 'bar-7']:
        for alg in ['gosdt', 'pygosdt']:
            performance(data, alg)

elif command == "compas":
    dataframe = pd.DataFrame(pd.read_csv("experiments/scalability/compas.csv")).dropna()
    # This selects current age and age at first offense
    X = pd.DataFrame(dataframe.iloc[:,3:5], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])

    if sys.argv[2] == "probe":
        performance_continuous(command, X, y, 'gosdt')
    else:
        for alg in ['cart', 'binoct', 'dl85', 'gosdt', 'pygosdt', 'osdt']:
            performance_continuous(command, X, y, alg)

elif command == "fico":
    dataframe = pd.DataFrame(pd.read_csv("experiments/scalability/fico.csv")).dropna()
    # This selects PercentTradesWBalance and ExternalRiskEstimate
    X = pd.DataFrame(dataframe.iloc[:,0:2], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])

    if sys.argv[2] == "probe":
        performance_continuous(command, X, y, 'gosdt')
    else:
        for alg in ['cart', 'dl85', 'gosdt', 'pygosdt', 'osdt']:
            performance_continuous(command, X, y, alg)

elif command == "coupon":
    dataframe = pd.DataFrame(pd.read_csv("experiments/scalability/coupon.csv")).dropna()
    # This selects all columns
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])

    if sys.argv[2] == "probe":
        performance_continuous(command, X, y, 'gosdt')
    else:
        for alg in ['cart', 'dl85', 'gosdt', 'pygosdt', 'osdt']:
            performance_continuous(command, X, y, alg)

elif command == "compas-2016":
    dataframe = pd.DataFrame(pd.read_csv("experiments/scalability/compas-2016.csv")).dropna()
    # This selects all columns
    X = pd.DataFrame(dataframe.iloc[:,:-1], columns=dataframe.columns[:-1])
    y = pd.DataFrame(dataframe.iloc[:,-1], columns=dataframe.columns[-1:])

    if sys.argv[2] == "probe":
        performance_continuous(command, X, y, 'gosdt')
    else:
        for alg in ['cart', 'dl85', 'gosdt', 'pygosdt', 'osdt']:
            performance_continuous(command, X, y, alg)

elif command == "profile":
    for data in ['car-evaluation', 'compas-binary', 'fico-binary', 'monk1-train', 'monk2-train', 'monk3-train', 'tic-tac-toe', 'bar-7']:
        for alg in ['osdt', 'gosdt', 'pygosdt']:
            profile(data, alg)

elif command == "imbalance":
    for data in ['tic-tac-toe']:
        for alg in ['gosdt-balanced', 'gosdt-f1', 'gosdt-auc']:
            imbalance(data, alg)