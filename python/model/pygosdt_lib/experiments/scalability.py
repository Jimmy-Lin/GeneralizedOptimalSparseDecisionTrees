import numpy as np
import matplotlib.pyplot as plt

from gc import collect
from time import time, sleep
from math import floor
from mpl_toolkits import mplot3d

from model.pygosdt_lib.experiments.logger import Logger
from model.pygosdt_lib.data_structures.dataset import read_dataframe


def scalability_analysis(dataset, model_class, hyperparameters, path, step_count=10):
    X = dataset.values[:, :-1]
    Y = dataset.values[:, -1]
    (n, m) = X.shape
    sample_size_step = max(1, round(n / step_count))
    feature_size_step = max(1, round(m / step_count))

    logger = Logger(path=path, header=['samples', 'features', 'runtime'])

    for sample_size in range(1, n+1, sample_size_step):
        for feature_size in range(1, m+1, feature_size_step):
            print("Subsample Shape: ({}, {})".format(sample_size, feature_size))

            # Try to standardize starting state
            collect()
            sleep(1)

            # Take Subsample
            x = X[:sample_size, :feature_size]
            y = Y[:sample_size]

            model = model_class(**hyperparameters)
            start = time()
            reruns = 5
            runtimes = []
            for i in range(reruns):
                try:
                    model.fit(x, y)
                except Exception as e:
                    print(e)
                    pass
                runtime = time() - start
                runtimes.append(runtime)

            runtime = sorted(runtimes)[floor(reruns/2)]
            logger.log([sample_size, feature_size, runtime])


def plot_scalability_analysis(dataset, title, z_limit=None):
    (n, m) = dataset.shape
    x = list(sorted(set(dataset.values[:, 0])))
    y = list(sorted(set(dataset.values[:, 1])))
    Y, X = np.meshgrid(y, x)
    Z = np.array(dataset.values[:, 2]).reshape(len(x), len(y))

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('Sample Size N')
    ax.set_ylabel('Feature Size M')
    ax.set_zlabel('Runtime (s)')
    ax.set_title(title)

    if z_limit != None:
        ax.set_zlim(0, z_limit)

    ax.view_init(50, -20)
