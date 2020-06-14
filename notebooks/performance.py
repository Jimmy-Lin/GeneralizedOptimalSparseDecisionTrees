import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits import mplot3d
from math import log
import pandas as pd
import numpy as np
import seaborn as sns
import re
plt.rc('font', size=10)
colors = ['#509eba', '#e3a042', '#d77ede', '#233c82', '#613717']
markers = ['o', 's', 'D', 'v', '^',]



for dataset in ['car-evaluation', 'compas-binary', 'fico-binary', 'monk-1', 'monk-2', 'monk-3', 'tic-tac-toe', 'bar-7']:
    # for selector in ['Test Accuracy', 'Training Accuracy']:
    #     plt.figure(figsize=(16, 10), dpi=80)
    #     figure, axes = plt.subplots(nrows=2, ncols=2)
    #     figure.tight_layout(pad=1)
    #     models = {}
    #     alpha = 1.0
    #     beta = 1.0
    #     gamma = 1.0
    #     score_function = lambda accuracy, complexity, duration : alpha * accuracy - beta * complexity - gamma * duration

    for conf_index in range(7):
        plt.figure(figsize=(16, 10), dpi=80)
        figure, axes = plt.subplots(nrows=2, ncols=2)
        figure.tight_layout(pad=1)

        for y_axis, y_label, plot_x, plot_y in [
            ('# Leaves', '# Leaves', 0, 0), 
            ('Training Time', 'Training Time (s)', 0, 1),
            ('Training Accuracy', 'Training Accuracy (%)', 1, 0),
            ('Test Accuracy', 'Test Accuracy (%)', 1, 1)]:

            results = pd.DataFrame(pd.read_csv('../experiments/results/performance_{}.csv'.format(dataset)))

            algorithms = list(sorted(set(results['Algorithm'])))
            present_algorithms = []
            CTEs = []
            error = []
            lows = []
            highs = []
            models = {}

            for i, algorithm in enumerate(algorithms):
                subresults = results[results['Algorithm']==algorithm]
                configurations = list(
                    sorted(
                        set(zip(subresults['Depth Limit'], subresults['Width Limit'], subresults['Regularization'])),
                    )
                )
                subresults = subresults[subresults['Training Time'] > 0][subresults['Training Time'] < 300]

                config = configurations[conf_index]

                # best_score = 0
                # best_config = None
                # best_centroid = 0
                # best_low = 0
                # best_high = 0
                # best_tex = None
                # best_points = None

                # for config in configurations:
                (depth, width, reg) = config
                points = subresults[subresults['Depth Limit']==depth]
                points = points[points['Width Limit']==width]
                points = points[points['Regularization']==reg]
                if len(points) <= 0:
                    continue
                # compute the y-median
  
                # score = np.median(points[''])
                scoring_tree = sorted(points['Latex'].tolist(), key=lambda tree: len(tree))[0]

                # if score > best_score:
                # best_score = score

                best_points = list(points[y_axis])
                best_centroid = np.median(points[y_axis])
                iqr = points[y_axis].quantile([0.25, 0.5, 0.75])
                best_low = iqr[0.5]-iqr[0.25]
                best_high = iqr[0.75]-iqr[0.5]
                best_config = config

                best_tex = scoring_tree[3]
                
                # if not best_config is None:
                    # print(dataset, algorithm, best_config,  best_score, y_axis, best_points)

                present_algorithms.append(algorithm)
                models[algorithm] = best_tex
                CTEs.append(best_centroid)
                lows.append(best_low)
                highs.append(best_high)

            x_pos = list( x * 1 for x in np.arange(len(present_algorithms)))
            axes[plot_x, plot_y].bar(x_pos, CTEs, yerr=[lows, highs], align='center', alpha=0.5, ecolor='black', capsize=5)
            axes[plot_x, plot_y].set_ylabel(y_label)
            axes[plot_x, plot_y].set_xticks(x_pos)
            axes[plot_x, plot_y].set_xticklabels(present_algorithms)
            axes[plot_x, plot_y].set_title('{}'.format(y_axis))
            # print("plot coords ", plot_x, plot_y)

        # Save the figure and show
        # plt.title('Model Performance (Configured for Max {})'.format(selector))
        plt.savefig('performance_{}_conf_{}.png'.format(dataset, conf_index).replace(" ", "_").lower())
        plt.clf()

        f = open("performance_{}_conf_{}.tex".format(dataset, conf_index).replace(" ", "_").lower(),"w")
        for algorithm, tex in models.items():
            f.write(algorithm + "\n")
            f.write(tex + "\n")
        f.close()