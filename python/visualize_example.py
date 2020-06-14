# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Example
# Example data set training with demonstration of visualizations:
#  - Text Visualization of Boundary Logic and Resulting Conditional Probability Distribution
#  - Graphical Visualization of 2D Classifier
#  - Graphical Visualization of 2D Classifier's Prediction Confidence

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from model.gosdt import GOSDT
from model.pygosdt import PyGOSDT
from model.osdt import OSDT
from model.visualizer import plot2DClassifier, plot2DConfidence

# dataframe = pd.DataFrame(pd.read_csv('experiments/preprocessed/compas-binary.csv'))
# dataframe = pd.DataFrame(pd.read_csv('experiments/datasets/compas/train.csv'))
# dataframe = pd.DataFrame(pd.read_csv('experiments/datasets/anon/train.csv'))
# dataframe = pd.DataFrame(pd.read_csv('experiments/datasets/iris/original.csv'))
dataframe = pd.DataFrame(pd.read_csv('experiments/scalability/iris-setosa.csv'))

X = dataframe[dataframe.columns[0:2]]
y = dataframe[dataframe.columns[-1:]]

hyperparameters = {
    # "objective": "acc",
    "regularization": 0.01,
    "uncertainty_tolerance": 0.0,
    "output_limit": 1,
    "time_limit": 30000,

    "verbose": True,

    "optimism": 0.7,
    "equity": 0.5,
    "sample_depth": 1,

    "opencl_platform_index": -1,
    "opencl_device_index": -1
}

model = OSDT(hyperparameters, preprocessor="complete")
start = time.time()
model.fit(X, y)
finish = time.time()
print("Execution Time: {}".format(finish - start))

prediction = model.predict(X)
training_accuracy = model.score(X, y)
print("Training Accuracy: {}".format(training_accuracy))
print(model.tree)
plot2DClassifier("Visualizer", X, y, model)

# %% [markdown]
# ### 2D Classifer Visualization

# %%
plot2DClassifier('GOSDT with TreeEncoding', X, y, model)

# %% [markdown]
# ### 2D Classifer Visualization (Confidence Levels)

# %%
for i, tree in enumerate(forest.trees):
    if i >= 10:
        print("Too many models produced, truncating visualization")
        break
    plot2DConfidence("Tree {} (Classification Confidence)".format(i), X, y, tree)


# %%


