import numpy as np
import pandas as pd
import math
import time
import pickle
from itertools import product, compress
from gmpy2 import mpz
import sklearn.tree
import sklearn.metrics

from .rule import make_all_ones, rule_vand, rule_vectompz, count_ones

def log(tic, lines, COUNT_POP, COUNT, queue, metric, R_c, tree_old, tree_new, sorted_new_tree_rules, fname):
    "log"

    the_time = str(time.time() - tic)

    the_count_pop = str(COUNT_POP)
    the_count = str(COUNT)
    the_queue_size = str(0)  # str(len(queue))
    the_metric = str(metric)
    the_Rc = str(R_c)

    the_old_tree = str(0)  # str(sorted([leaf.rules for leaf in tree_old.cache_tree.leaves]))
    the_old_tree_splitleaf = str(0)  # str(tree_old.splitleaf)
    the_old_tree_objective = str(tree_old.cache_tree.risk)
    the_old_tree_lbound = str(tree_old.lb)
    the_new_tree = str(0)  # str(list(sorted_new_tree_rules))
    the_new_tree_splitleaf = str(0)  # str(tree_new.splitleaf)

    the_new_tree_objective = str(0)  # str(tree_new.cache_tree.risk)
    the_new_tree_lbound = str(tree_new.lb)
    the_new_tree_length = str(0)  # str(len(tree_new.cache_tree.leaves))
    the_new_tree_depth = str(0)  # str(max([len(leaf.rules) for leaf in tree_new.leaves]))

    the_queue = str(0)  # str([[ leaf.rules for leaf in thetree.leaves]  for _,thetree in queue])

    line = ";".join([the_time, the_count_pop, the_count, the_queue_size, the_metric, the_Rc,
                     the_old_tree, the_old_tree_splitleaf, the_old_tree_objective, the_old_tree_lbound,
                     the_new_tree, the_new_tree_splitleaf,
                     the_new_tree_objective, the_new_tree_lbound, the_new_tree_length, the_new_tree_depth,
                     the_queue
                     ])

    with open(fname, 'a+') as f:
        f.write(line+'\n')
        
def gini_reduction(x_mpz, y_mpz, n, rule_idx, points_cap=None):
    """
    calculate the gini reduction by each feature
    return the rank of by descending
    """
    if points_cap == None:
        points_cap = make_all_ones(n + 1)

    ndata0 = count_ones(points_cap)
    _, ndata01 = rule_vand(y_mpz, points_cap)

    p0 = ndata01 / ndata0
    gini0 = 2 * p0 * (1 - p0)

    gr = []
    for i in rule_idx:
        xi = x_mpz[i]
        l1_cap, ndata1 = rule_vand(points_cap, ~xi | mpz(pow(2, n)))

        _, ndata11 = rule_vand(l1_cap, y_mpz)

        l2_cap, ndata2 = rule_vand(points_cap, xi)

        _, ndata21 = rule_vand(l2_cap, y_mpz)

        p1 = ndata11 / ndata1 if ndata1 != 0 else 0
        p2 = ndata21 / ndata2 if ndata2 != 0 else 0
        gini1 = 2 * p1 * (1 - p1)
        gini2 = 2 * p2 * (1 - p2)
        gini_red = gini0 - ndata1 / ndata0 * gini1 - ndata2 / ndata0 * gini2
        gr.append(gini_red)

    gr = np.array(gr)
    order = list(gr.argsort()[::-1])

    odr = [rule_idx[r] for r in order]

    '''
    print("gr:", gr)
    print("order:", order)
    print("odr:", odr)
    '''

    dic = dict(zip(np.array(rule_idx)+1, odr))

    return odr, dic


def get_code(tree, feature_names, target_names, spacer_base="    "):
    """Produce psuedo-code for scikit-leant DescisionTree.
        Args
        ----
        tree -- scikit-leant DescisionTree.
        feature_names -- list of feature names.
        target_names -- list of target (class) names.
        spacer_base -- used for spacing code (default: "    ").
        Notes
        -----
        based on http://stackoverflow.com/a/30104792.
        http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
        """
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    feats = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print((spacer + "if ( " + feats[node] + " <= " + str(threshold[node]) + " ) {"))
            if left[node] != -1:
                recurse(left, right, threshold, feats, left[node], depth + 1)
            print((spacer + "}\n" + spacer + "else {"))
            if right[node] != -1:
                recurse(left, right, threshold, feats, right[node], depth + 1)
            print((spacer + "}"))
        else:
            target = value[node]
            print((spacer + "return " + str(target)))
            for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print((spacer + "return " + str(target_name) + " " + str(i) + " " \
                                                                              " ( " + str(
                    target_count) + " examples )"))

    recurse(left, right, threshold, feature_names, 0, 0)
    
def cart(x, y, name, n, P, N, lamb, w, theta, MAXDEPTH):
    clf = sklearn.tree.DecisionTreeClassifier(
            max_depth=None if MAXDEPTH == float('Inf') else MAXDEPTH,
            min_samples_split=max(math.ceil(lamb * 2 * n), 2),
            min_samples_leaf=math.ceil(lamb * n),
            max_leaf_nodes=math.floor(1 / (2 * lamb)),
            min_impurity_decrease=lamb)
    clf = clf.fit(x, y)
    nleaves_CART = (clf.tree_.node_count + 1) / 2
    pred_CART = clf.predict(x)
    if name == 'acc':
        trainout_CART = sklearn.metrics.accuracy_score(y, pred_CART)
    elif name == "bacc":
        trainout_CART = sklearn.metrics.balanced_accuracy_score(y, pred_CART)
    elif name == 'wacc':
        n_fp = sum((pred_CART == 1) & (pred_CART != y))
        n_fn = sum((pred_CART == 0) & (pred_CART != y))
        trainout_CART = 1-(n_fp + w*n_fn)/(w*P+N)
    elif name == 'f1':
        trainout_CART = sklearn.metrics.f1_score(y, pred_CART)
    elif name == 'auc_convex':
        nodes = clf.apply(x)
        metric = np.empty([3, int(nleaves_CART)], dtype=float)
        for i, num in enumerate(set(nodes)):
            idx = np.where(nodes == num)
            metric[0,i] = sum(y[idx]==1)
            metric[1,i] = sum(y[idx]==0)
            metric[2,i] = sum(y[idx]==1)/len(y[idx])
        metric = metric[:,np.argsort(metric[2,])]
        metric = np.flip(metric, axis=1)
        metric = np.cumsum(metric,axis=1)
        init = np.array([[0], [0], [0]])
        metric = np.append(init, metric, axis=1)
        trainout_CART = 0.5*sum([(metric[0,i]/P+metric[0,i-1]/P)*(metric[1,i]/N-metric[1,i-1]/N) for i in range(1,int(nleaves_CART)+1)])
    
    elif name == 'partial_auc':
        nodes = clf.apply(x)
        metric = np.empty([3, int(nleaves_CART)], dtype=float)
        for i, num in enumerate(set(nodes)):
            idx = np.where(nodes == num)
            metric[0,i] = sum(y[idx]==1)
            metric[1,i] = sum(y[idx]==0)
            metric[2,i] = sum(y[idx]==1)/len(y[idx])
        metric = metric[:,np.argsort(metric[2,])]
        metric = np.flip(metric, axis=1)
        metric = np.cumsum(metric,axis=1)
        init = np.array([[0], [0], [0]])
        metric = np.append(init, metric, axis=1)
        
        tp = [0]
        fp = [0]
        i = 1
        while fp[i-1] < N*theta and i <= int(nleaves_CART)+1:
            tp.append(metric[0,i])
            fp.append(metric[1,i])
            i += 1
            
        tp[i-1] = ((tp[i-1]-tp[i-2])/(fp[i-1]-fp[i-2]))*(N*theta-fp[i-1])+tp[i-1]
        fp[i-1] = N*theta
        trainout_CART = 0.5*sum([(tp[i]/P+tp[i-1]/P)*(fp[i]/N-fp[i-1]/N) for i in range(1,len(tp))])
     
        

    R_c = 1 - trainout_CART + lamb*nleaves_CART
    d_c = clf
    C_c = 0
    return clf, nleaves_CART, trainout_CART, R_c, d_c, C_c

def get_z(x, y, n, m):
    """
    calculate z, which is for the equivalent points bound
    z is the vector defined in algorithm 5 of the CORELS paper
    z is a binary vector indicating the data with a minority lable in its equivalent set
    """
    z = pd.DataFrame([-1] * n).values
    # enumerate through theses samples
    for i in range(n):
        # if z[i,0]==-1, this sample i has not been put into its equivalent set
        if z[i, 0] == -1:
            tag1 = np.array([True] * n)
            for j in range(m):
                rule_label = x[i][j]
                # tag1 indicates which samples have exactly the same features with sample i
                tag1 = (x[:, j] == rule_label) * tag1

            y_l = y[tag1]
            pred = int(y_l.sum() / len(y_l) >= 0.5)
            # tag2 indicates the samples in a equiv set which have the minority label
            tag2 = (y_l != pred)
            z[tag1, 0] = tag2

    z_mpz = rule_vectompz(z.reshape(1, -1)[0])
    return z_mpz
