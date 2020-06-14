import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot2DClassifier(title, X, y, model):
    (n,m) = X.shape
    y_hat = model.predict(X).tolist()
    correct= [  y.iloc[i,0] == y_hat[i] for i in range(n) ]
    fig = plt.figure(figsize=(8,8), dpi=100)
    labels = list(set(y.values.reshape(1,n)[0].tolist()))
    colormap = ['red', 'green', 'blue']
    for k, label in enumerate(labels):
        idx_correct = np.array([ int(i) for c, i in zip(correct, (y == label).values.reshape(1,n)[0]) ], dtype=bool)

        idx_correct = np.array([ int(i and c) for c, i in zip(correct, (y == label).values.reshape(1,n)[0]) ], dtype=bool)
        idx_incorrect = np.array([ int(i and not c) for c, i in zip(correct, (y == label).values.reshape(1,n)[0]) ], dtype=bool)
        x_domain = X[idx_correct][X.columns[0]]
        y_domain = X[idx_correct][X.columns[1]]
        plt.plot(x_domain, y_domain,
            label="{} = {} (correct)".format(y.columns[0], str(label)),
            markersize=7.5,
            marker='.',
            markerfacecolor=colormap[k],
            color=colormap[k],
            linewidth=0,
            markeredgewidth=1)
        x_domain = X[idx_incorrect][X.columns[0]]
        y_domain = X[idx_incorrect][X.columns[1]]
        plt.plot(x_domain, y_domain,
            label="{} = {} (incorrect)".format(y.columns[0], str(label)),
            markersize=7.5,
            marker='x',
            markerfacecolor=colormap[k],
            color=colormap[k],
            linewidth=0,
            markeredgewidth=1)

    steps = 250
    x_width = max(X.values[:,0]) - min(X.values[:,0])
    y_width = max(X.values[:,1]) - min(X.values[:,1])
    x_lim = (min(X.values[:,0]) - 0.05 * x_width, max(X.values[:,0]) + 0.05 * x_width)
    y_lim = (min(X.values[:,1]) - 0.05 * y_width, max(X.values[:,1]) + 0.05 * y_width)
    x_domain = np.linspace(x_lim[0], x_lim[1], steps)
    y_domain = np.linspace(y_lim[0], y_lim[1], steps)
    xv, yv = np.meshgrid(x_domain, y_domain)

    xy = []
    for x in x_domain:
        for y in y_domain:
            xy.append([x,y])
    xy = pd.DataFrame(np.array(xy))
    z_raw = model.predict(xy)
    z = [0 for i in range(steps * steps)]
    for i in range(steps * steps):
        z[i] = labels.index(z_raw[i])
    z = np.array(z).reshape(steps, steps).transpose()

    plt.contourf(xv, yv, z, [ i - 0.5 for i in range(len(labels) + 1)], colors=colormap, alpha=0.5)

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)         # ticks along the top edge are off
    
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False)         # ticks along the top edge are off

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.show()

def plot2DConfidence(title, X, y, model):
    (n,m) = X.shape
    fig = plt.figure(figsize=(8,8), dpi=100)

    labels = list(set(y.values.reshape(1,n)[0].tolist()))
    colormap = ['red', 'green', 'blue']
    for k, label in enumerate(labels):
        idx = np.array([int(i) for i in (y == label).values.reshape(1,n)[0]], dtype=bool)
        x_domain = X[idx][X.columns[0]]
        y_domain = X[idx][X.columns[1]]
        plt.plot(x_domain, y_domain,
            label="{} = {}".format(y.columns[0], str(label)),
            markersize=7.5,
            marker='.',
            markerfacecolor=colormap[k],
            linewidth=0,
            markeredgewidth=0.5)

    steps = 250
    x_width = max(X.values[:,0]) - min(X.values[:,0])
    y_width = max(X.values[:,1]) - min(X.values[:,1])
    x_lim = (min(X.values[:,0]) - 0.05 * x_width, max(X.values[:,0]) + 0.05 * x_width)
    y_lim = (min(X.values[:,1]) - 0.05 * y_width, max(X.values[:,1]) + 0.05 * y_width)
    x_domain = np.linspace(x_lim[0], x_lim[1], steps)
    y_domain = np.linspace(y_lim[0], y_lim[1], steps)
    xv, yv = np.meshgrid(x_domain, y_domain)

    xy = []
    for x in x_domain:
        for y in y_domain:
            xy.append([x,y])
    xy = pd.DataFrame(np.array(xy))
    z = model.confidence(xy).reshape(steps, steps).transpose()

    # fig = plt.figure(figsize=(8, 6), dpi=100)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(yv, xv, z, rstride=1, cstride=1,
    #                 cmap='viridis', edgecolor='none')
    # ax.set_ylabel(X.columns[0])
    # ax.set_xlabel(X.columns[1])
    # ax.set_zlabel('Confidence Level')
    # ax.set_title('Confidence vs Feature Space')

    # ax.view_init(alpha, beta)
        
    plt.contourf(xv, yv, z, vmin=0.0, vmax=1.0, cmap=plt.get_cmap('Greys'), alpha=0.9)

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.grid()
    plt.legend()
    plt.title(title)


# def plot2DRegressor(title, X, y, model):
#     (n,m) = X.shape
#     fig = plt.figure(figsize=(8,8), dpi=100)
#     labels = list(set(y.values.reshape(1,n)[0].tolist()))
#     colormap = ['red', 'green', 'blue']
#     for k, label in enumerate(labels):
#         idx = np.array([int(i) for i in (y == label).values.reshape(1,n)[0]], dtype=bool)
#         x_domain = X[idx][X.columns[0]]
#         y_domain = X[idx][X.columns[1]]
#         plt.plot(x_domain, y_domain,
#             label="{} = {}".format(y.columns[0], str(label)),
#             markersize=7.5,
#             marker='o',
#             markerfacecolor=colormap[k],
#             linewidth=0,
#             markeredgewidth=0.5)

#     steps = 250
#     x_width = max(X.values[:,0]) - min(X.values[:,0])
#     y_width = max(X.values[:,1]) - min(X.values[:,1])
#     x_lim = (min(X.values[:,0]) - 0.05 * x_width, max(X.values[:,0]) + 0.05 * x_width)
#     y_lim = (min(X.values[:,1]) - 0.05 * y_width, max(X.values[:,1]) + 0.05 * y_width)
#     x_domain = np.linspace(x_lim[0], x_lim[1], steps)
#     y_domain = np.linspace(y_lim[0], y_lim[1], steps)
#     xv, yv = np.meshgrid(x_domain, y_domain)

#     xy = []
#     for x in x_domain:
#         for y in y_domain:
#             xy.append([x,y])
#     xy = pd.DataFrame(np.array(xy))
#     z_raw = model.predict(xy)
#     z = [0 for i in range(steps * steps)]
#     for i in range(steps * steps):
#         z[i] = labels.index(z_raw[i])
#     z = np.array(z).reshape(steps, steps)

#     # plt.contourf(xv, yv, z.transpose(), colors=colormap, alpha=0.25)
#     plt.contourf(xv, yv, z.transpose(), [ i - 0.5 for i in range(len(labels) + 1)], colors=colormap, alpha=0.25)
#     # plt.contourf(xv, yv, z, [i for i in range(len(labels))], colors=colormap, alpha=0.25)

#     plt.xlabel(X.columns[0])
#     plt.ylabel(X.columns[1])
#     plt.grid()
#     plt.legend()
#     plt.title(title)