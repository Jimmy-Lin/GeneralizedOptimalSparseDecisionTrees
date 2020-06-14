import cplex
from cplex.exceptions import CplexError
import sys
import csv
from scipy.stats.stats import pearsonr
import math

inputstart = 0
inputsym = 0
inputtime = 900
inputpolish = 100

double_data = False

# The data, DATA_TABLE[-1] contains the target, others are features
DATA_TABLE = []
# The possible values of constants, a list of values per feature
CONSTANT_VALS = dict()
# MIN and MAX values for every feature, feature is an int
MIN_VALUE = dict()
MAX_VALUE = dict()
# MIN and MAX targets
TARGETS = []
# MINIMUM DISTANCE between feature values
MIN_DIST = 1.0
# USED BY input reading
MAX_VALUES = []

# VARIABLE NAMES FOR LP ENCODING
VARS = dict()


# Jason's code for obtaining decision boundaries from a Random Forest

# get number of parents of node until root
def get_num_parents(node, num_nodes):
    return get_num_parents_recur(((num_nodes+1) / 2) - 1, node, ((num_nodes+1) / 2) / 2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import numpy as np
import pandas as pd


# input is a decision tree object <dct>, the number of features in the dataset <num_features>
# and a list with the indexes (indexes start at 1) of features that you do not want cutoff of <indexes_binary>
def get_thresholds(dct, num_features, indexes_binary):
    cutoff_list = [[]]*num_features
    n_nodes = dct.tree_.node_count
    #children_left = dct.tree_.children_left
    #children_right = dct.tree_.children_right
    feature = dct.tree_.feature
    threshold = dct.tree_.threshold
    selected_features = []
    
    for ii in range(1, n_nodes+1, 1):
        test=feature[ii-1]
        
        # check if not in leaf node or binary feature
        if (test != -2 and test not in indexes_binary):
            test = test +  1
            add_cutoff =threshold[ii-1] 
            old= cutoff_list[test-1]
            new = old + [add_cutoff]
            cutoff_list[test-1] = new
            selected_features.append(test)
    
    selected_features = list(np.unique(selected_features))
    not_selected_features = []
    for ii in range(1, num_features +1 , 1):
        if (ii not in selected_features and ii not in indexes_binary):
            not_selected_features.append(ii)
    
    return cutoff_list, selected_features, not_selected_features

# input is a list of decision tree classifiers <Forest>
# output: list cutoffs, list of selected features, list of not selected features, dictionary of selected features + cutoffs
# cutoff_list_old[i] = list of cutoffs for feature i
# len(cutoff_list_old) = num_features

def get_thresholds_FOREST(Forest, num_features, indexes_binary):
    n_estimators = len(Forest)
    
    dct1 = Forest[0]
    cutoff_list_old, selected_features_old, not_selected_features = get_thresholds(dct1, num_features, indexes_binary)

    for ii in range(2,n_estimators + 1, 1 ):
        dct_add = Forest[ii-1]
        cutoff_list_add, selected_features_add, not_selected_features_add = get_thresholds(dct_add, num_features, indexes_binary)

        cutoff_list_old=update_cutoffs_FOREST(cutoff_list_old, cutoff_list_add)
        selected_features_old, not_selected_features=update_selected_features(selected_features_old, selected_features_add, num_features, indexes_binary)
    
    # make a dictionary
    cutoff_dict = dict(zip(range(1,num_features + 1, 1 ), cutoff_list_old))
    
    return cutoff_list_old, selected_features_old, not_selected_features, cutoff_dict


def update_cutoffs_FOREST(cutoff_list_old, cutoff_list_add):
    num_features = len(cutoff_list_old)
    cutoff_list_new = []
    for ii in range(1, num_features +1 , 1):
        add = cutoff_list_old[ii-1] + cutoff_list_add[ii-1]
        add = list(np.unique(add))
        cutoff_list_new.append( add)
        
    
    return cutoff_list_new


def update_selected_features(selected_features_old, selected_features_add, num_features, indexes_binary):
    selected_features_new = selected_features_old + selected_features_add
    selected_features_new = list(np.unique(selected_features_new))
    
    not_selected_features = []
    for ii in range(1, num_features +1 , 1):
        if (ii not in selected_features_new and ii not in indexes_binary):
            not_selected_features.append(ii)
    
    
    return selected_features_new, not_selected_features


# example using iris data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

forest = RandomForestClassifier(max_depth=3, n_estimators=1)
forest.fit(X_train, y_train)

num_features = 4
indexes_binary = []


cutoff_list_tree, selected_features_tree, not_selected_features_tree , cutoff_dict_tree= get_thresholds_FOREST([tree], num_features, indexes_binary)

cutoff_list_forest, selected_features_forest, not_selected_features_forest, cutoff_dict_forest = get_thresholds_FOREST(forest, num_features, indexes_binary)


print("dictionary with cutoffs for Forest...")
print(cutoff_dict_forest)



# OLD code from tree encoding, imported in order to use trees as starting solutions
# all functions starting with s get values from scikitlearn tree objects
def sget_feature(tree, index):
    return tree.tree_.feature[index]

def sget_leaf_constant(tree, index):
    return tree.tree_.value[index]

def sget_node_constant(tree, index):
    return tree.tree_.threshold[index]

def sget_left_node(tree, index):
    return tree.tree_.children_left[index]

def sget_right_node(tree, index):
    return tree.tree_.children_right[index]

def sget_parent_node(tree, index):
    if index == 0:
        return -1
    for i in range(len(tree.tree_.children_left)):
        if tree.tree_.children_left[i] == index:
            return i
    for i in range(len(tree.tree_.children_right)):
        if tree.tree_.children_right[i] == index:
            return index
    return -1

def sget_path(tree, index):
    if index == 0:
        return [], []
    for i in range(len(tree.tree_.children_left)):
        if tree.tree_.children_left[i] == index:
            path, truth_values = get_path(tree,i)
            path.append(str(i))
            truth_values.append(0)
            return path, truth_values
    for i in range(len(tree.tree_.children_right)):
        if tree.tree_.children_right[i] == index:
            path, truth_values = get_path(tree,i)
            path.append(str(i))
            truth_values.append(1)
            return path, truth_values
    return [], []

def sget_parent(tree, index):
    if index == 0:
        return ""
    for i in range(len(tree.tree_.children_left)):
        if tree.tree_.children_left[i] == index:
            return str(i) + "_T"
    for i in range(len(tree.tree_.children_right)):
        if tree.tree_.children_right[i] == index:
            return str(i) + "_F"
    return ""

def sget_children(tree, index):
    return tree.tree_.children_left[index], tree.tree_.children_right[index]

def sget_lower_leafs(tree, index):
    result = []
    left, right = get_children(tree, index)
    if left != -1:
        result = result + get_lower_leafs(tree, left)
        result = result + get_lower_leafs(tree, right)
    else:
        return [index]
    return result
    
def sget_left_leafs(tree, index):
    left, right = get_children(tree, index)
    if left != -1:
        return get_lower_leafs(tree, left)
    return []

def sget_right_leafs(tree, index):
    left, right = get_children(tree, index)
    if right != -1:
        return get_lower_leafs(tree, right)
    return []

def sget_bounds(tree, index):
    left, right = get_children(tree, index)
    leftub,leftlb,rightub,rightlb = 0.0,0.0,0.0,0.0
    if tree.tree_.children_left[left] != -1:
        leftlb,leftub = get_bounds(tree, left)
    else:
        leftlb,leftub = get_leaf_constant(tree,left),get_leaf_constant(tree,left)
        
    if tree.tree_.children_right[left] != -1:
        rightlb,rightub = get_bounds(tree, right)
    else:
        rightlb,rightub = get_leaf_constant(tree,right),get_leaf_constant(tree,right)

    return min(leftlb, rightlb), max(leftub,rightub)

def snode_lists(trees):
    leafs = dict()
    nodes = dict()
    for t in gen._ITEMS_:
        ls = []
        ns = []
        for i in range(len(trees[t].tree_.value)):
            if trees[t].tree_.children_left[i] != -1:
                ns.append(i)
            else:
                ls.append(i)
        leafs[t] = ls
        nodes[t] = ns
    return nodes, leafs

# functions for tree traversing and getting min/max/row values for features and the target

# get min values for feature f
def get_min_value_f(f):
    global MIN_VALUE
    if double_data == False:
        return MIN_VALUE[f]

    if f >= len(DATA_TABLE[0]) - 1:
        f = f-len(DATA_TABLE[0]) + 1
        return get_max_value() - MAX_VALUE[f]
    return MIN_VALUE[f]

# get min value over all features
def get_min_value():
    global MIN_VALUE
    return min(MIN_VALUE.values())

# get max value for feature f
def get_max_value_f(f):
    global MAX_VALUE
    
    if double_data == False:
        return MAX_VALUE[f]
    
    if f >= len(DATA_TABLE[0]) - 1:
        f = f-len(DATA_TABLE[0]) + 1
        return get_max_value() - MIN_VALUE[f]
    return MAX_VALUE[f]

# get max value over all features
def get_max_value():
    global MAX_VALUE
    return max(MAX_VALUE.values())

def get_max_target():
    return max(TARGETS)

def get_min_target():
    return min(TARGETS)

def get_max_error():
    return get_max_target() - get_min_target()

# get feature value f in row d
def get_feature_value(d,f):
    if double_data == False:
        return DATA_TABLE[d+1][f]
    
    if f < len(DATA_TABLE[0]) - 1:
        return DATA_TABLE[d+1][f]
    return get_max_value() - DATA_TABLE[d+1][f-len(DATA_TABLE[0])+1]

# get target in row d
def get_target(d):
    return DATA_TABLE[d+1][-1]

# get size of input table
def get_data_size():
    return len(DATA_TABLE) - 1

#get constant value
def get_constant_val(f,i):
    return CONSTANT_VALS[f][i]

def get_max_constant_val(f):
    if len(CONSTANT_VALS[f]) != 0:
        return max([CONSTANT_VALS[f][i] for i in range(len(CONSTANT_VALS[f]))])
    return get_max_value_f(f) - 1

def get_min_constant_val(f):
    if len(CONSTANT_VALS[f]) != 0:
        return min([CONSTANT_VALS[f][i] for i in range(len(CONSTANT_VALS[f]))])
    return get_min_value_f(f) + 1

# get number of constant values
def get_num_constants(f):
    return len(CONSTANT_VALS[f])

# get number of features
def get_num_features():
    if double_data == False:
        return len(DATA_TABLE[0]) - 1
    
    return 2*(len(DATA_TABLE[0]) - 1)

# get max number of constant values
def get_max_num_constants():
    return max([get_num_constants(f) for f in range(get_num_features())])

# get number of targets
def get_num_targets():
    return len(TARGETS)

# get feature name for feature f
def get_feature(f):
    if double_data == False:
        return DATA_TABLE[0][f]
    
    if f < len(DATA_TABLE[0]) - 1:
        return DATA_TABLE[0][f]
    return "inv" + DATA_TABLE[0][f-len(DATA_TABLE[0]) + 1]

def get_min_dist():
    return MIN_DIST

# get d'th feature value, sorted by increasing value, used to sorting function
SORTED_FEATURE = 0
def get_sorted_feature_value(d):
    global SORTED_FEATURE
    return get_feature_value(d, SORTED_FEATURE)

# get all unique targets, sorted by increasing value
def get_sorted_targets():
    return list(sorted(set([DATA_TABLE[d+1][-1] for d in range(get_data_size())])))

# get all unique feature values for feature f, sorted by increasing value
def get_sorted_feature_values(f):
    if double_data == False:
        return list(sorted(set([DATA_TABLE[d+1][f] for d in range(get_data_size())])))
    
    if f < len(DATA_TABLE[0]) - 1:
        return list(sorted(set([DATA_TABLE[d+1][f] for d in range(get_data_size())])))
    return list(sorted(set([get_max_value() - DATA_TABLE[d+1][f - len(DATA_TABLE[0]) + 1] for d in range(get_data_size())])))

# read data, set min max values, put into DATA_TABLE
def read_file(file_name):
    global DATA_TABLE, MIN_VALUE, MAX_VALUE, MIN_TARGET, MAX_TARGET, MIN_DIST
    DATA_TABLE = []
    MIN_VALUE = dict()
    MAX_VALUE = dict()
    MIN_DIST = 1.0

    data = []
    header = True
    with open(file_name, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if header is False:
                row = [float(i) for i in row]
                if row[-1] not in TARGETS:
                    TARGETS.append(row[-1])
                for f in range(len(row) - 1):
                    if f not in MIN_VALUE or MIN_VALUE[f] > row[f]:
                        MIN_VALUE[f] = row[f]
                    if f not in MAX_VALUE or MAX_VALUE[f] < row[f]:
                        MAX_VALUE[f] = row[f]

            header = False
            data.append(row)

    DATA_TABLE = data
    
    MIN_DIST = 1.0
    for index in range(len(DATA_TABLE[0])-1):
        values = []
        for d in range(1,len(DATA_TABLE)):
            values.append(DATA_TABLE[d][index])
        values = sorted(set(values))
        for d in range(1,len(values)-1):
            if MIN_DIST > (values[d+1] - values[d]):
                MIN_DIST = (values[d+1] - values[d])

    MIN_DIST = MIN_DIST * 0.5

# ONLY ROWS IN row_list, read data, set min max values, put into DATA_TABLE
def read_file_rows(file_name, row_set):
    global DATA_TABLE, MIN_VALUE, MAX_VALUE, MIN_TARGET, MAX_TARGET, MIN_DIST
    DATA_TABLE = []
    MIN_VALUE = dict()
    MAX_VALUE = dict()
    MIN_DIST = 1.0

    data = []
    header = True
    
    row_nr = 0
    
    with open(file_name, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if header is False:
                row_nr = row_nr + 1
                if row_nr-1 not in row_set:
                    continue
            
                row = [float(i) for i in row]
                if row[-1] not in TARGETS:
                    TARGETS.append(row[-1])
                for f in range(len(row) - 1):
                    if f not in MIN_VALUE or MIN_VALUE[f] > row[f]:
                        MIN_VALUE[f] = row[f]
                    if f not in MAX_VALUE or MAX_VALUE[f] < row[f]:
                        MAX_VALUE[f] = row[f]

            header = False
            data.append(row)

    DATA_TABLE = data
    
    MIN_DIST = 1.0
    for index in range(len(DATA_TABLE[0])-1):
        values = []
        for d in range(1,len(DATA_TABLE)):
            values.append(DATA_TABLE[d][index])
        values = sorted(set(values))
        for d in range(1,len(values)-1):
            if MIN_DIST > (values[d+1] - values[d]):
                MIN_DIST = (values[d+1] - values[d])

    MIN_DIST = MIN_DIST * 0.5

# write data
def write_file(file_name):
    global DATA_TABLE, MIN_VALUE, MAX_VALUE, MIN_TARGET, MAX_TARGET
    data = []
    header = True
    with open(file_name, 'wt') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        for d in range(len(DATA_TABLE)):
            writer.writerow(DATA_TABLE[d])

# transform data to successive integers, centered around 0
def transform_data():
    global MIN_VALUE, MAX_VALUE, DATA_TABLE, MAX_VALUES, CORRS, MIN_DIST, SORTED_FEATURE

    CORRS = dict()
    targets = []
    for d in range(1,len(DATA_TABLE)):
        targets.append(DATA_TABLE[d][-1])
    
    # scale to 0.0-1.0, not needed
    for index in range(len(DATA_TABLE[0])-1):
        for d in range(1,len(DATA_TABLE)):
            DATA_TABLE[d][index] = float(DATA_TABLE[d][index] - get_min_value_f(index)) / float(get_max_value_f(index) - get_min_value_f(index))

    # assign integers from 1 to the number of unique values
    MIN_DIST = 1.0
    for index in range(len(DATA_TABLE[0])-1):
        MAX_VALUE[index] = 1.0
        MIN_VALUE[index] = 0.0
        values = []
        for d in range(1,len(DATA_TABLE)):
            values.append(DATA_TABLE[d][index])
        values = sorted(set(values))
        for d in range(1,len(values)-1):
            if MIN_DIST > (values[d+1] - values[d]):
                MIN_DIST = (values[d+1] - values[d])

        for d in range(1,len(DATA_TABLE)):
            for v in range(len(values)):
                if values[v] == DATA_TABLE[d][index]:
                    DATA_TABLE[d][index] = float(v)
                    break

        MAX_VALUE[index] = float(len(values)) - 1.0
        MAX_VALUES.append(float(len(values)) - 1.0)
        MIN_VALUE[index] = 0.0
    MIN_DIST = 1.0
    
    MAX_VALUES = []

    # checks if successive occurrences of a feature-value have the same target-value
    # if all occurrences of two successive values have the same target, they are combined
    for index in range(len(DATA_TABLE[0])-1):
        SORTED_FEATURE = index
        all_rows = sorted(list(range(len(DATA_TABLE)-1)), key=get_sorted_feature_value)
        #print zip([get_feature_value(d,index) for d in all_rows],[get_target(d) for d in all_rows])
        
        cut_points = []
        previous_target = -1000
        previous_value = -1000
        for d in all_rows:
            if previous_target != get_target(d):
                cut_points.extend([get_feature_value(d,index) - 0.5])
            if previous_target != get_target(d) and previous_value == get_feature_value(d,index):
                cut_points.extend([get_feature_value(d,index) + 0.5])
            previous_target = get_target(d)
            previous_value = get_feature_value(d,index)
        cut_points.extend([len(DATA_TABLE)])
        cut_points = list(sorted(set(cut_points)))

        previous_cut_point = -1000
        data_copy = [DATA_TABLE[d][index] for d in range(len(DATA_TABLE))]
        for v in range(len(cut_points)):
            for d in range(1,len(DATA_TABLE)):
                if cut_points[v] > data_copy[d] and previous_cut_point < data_copy[d]:
                    DATA_TABLE[d][index] = float(v)
            previous_cut_point = cut_points[v]

        MAX_VALUE[index] = float(len(cut_points)) - 1.0
        MAX_VALUES.append(float(len(cut_points)) - 1.0)
        MIN_VALUE[index] = 0.0
        all_rows = sorted(list(range(len(DATA_TABLE)-1)), key=get_sorted_feature_value)
        MIN_VALUE[index] = get_feature_value(all_rows[0], index)
        MAX_VALUE[index] = get_feature_value(all_rows[len(all_rows)-1], index)
        #print zip([get_feature_value(d,index) for d in all_rows],[get_target(d) for d in all_rows])
        #print cut_points, MIN_VALUE[index], MAX_VALUE[index]

    # translate all feature value to be centered around 0
    for d in range(1,len(DATA_TABLE)):
        for index in range(len(DATA_TABLE[0])-1):
            maxv = MAX_VALUE[index]
            DATA_TABLE[d][index] = DATA_TABLE[d][index] - int(float(maxv)/2.0)
    for index in range(len(DATA_TABLE[0])-1):
        maxv = MAX_VALUE[index]
        MIN_VALUE[index] = MIN_VALUE[index] - int(float(maxv)/2.0)
        MAX_VALUE[index] = MAX_VALUE[index] - int(float(maxv)/2.0)

        #print MIN_VALUE[index], MAX_VALUE[index]

    return

# set possible constant values
def find_constants():
    global CONSTANT_VALS, MIN_VALUE, MAX_VALUE, MAX_VALUES, CORRS, MIN_DIST, SORTED_FEATURE

    # checks if successive occurrences of a feature-value have the same target-value
    # if all occurrences of two successive values have the same target,
    # there is no constant value in between
    CONSTANT_VALS = dict()
    
    for index in range(len(DATA_TABLE[0])-1):
        val_targets = dict()
        for d in range(get_data_size()):
            if get_feature_value(d, index) not in val_targets:
                val_targets[get_feature_value(d, index)] = set([get_target(d)])
            else:
                val_targets[get_feature_value(d, index)].add(get_target(d))
    
        cut_points = []#get_min_value(), get_max_value()]
        prev_list = []
        prev_key = -1
        for key in sorted(val_targets.keys()):
            list = val_targets[key]
            cut_point = float(prev_key + key) / 2.0
            
            if prev_list != []:
                if len(list) != 1 or len(prev_list) != 1:
                    cut_points.append(cut_point)
                elif list != prev_list:
                    cut_points.append(cut_point)
        
            prev_list = list
            prev_key = key

        #print val_targets, sorted(val_targets.iterkeys()), cut_points
        CONSTANT_VALS[index] = cut_points

    #for f in range(get_num_features()):
    #    print [get_feature_value(d,f) for d in range(get_data_size())]
    #    print CONSTANT_VALS[f]

def clear_constants():
    global CONSTANT_VALS
    for i in range(len(CONSTANT_VALS)):
        #print(CONSTANT_VALS[i])
        #print(cutoff_dict_forest[i+1])
        CONSTANT_VALS[i] = []
        
def add_constants_from_tree(depth):
    global CONSTANT_VALS
    dat = np.array(DATA_TABLE)
    #print(dat)
    x = pd.DataFrame(data=dat[1:,0:-1])
    #print(x)
    y = pd.DataFrame(data=dat[1:,-1])
    #print(y)

    # X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.0,random_state=0)
    X_train = X_test = x
    y_train = y_test = y
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(X_train, y_train)

    cutoff_list_forest, selected_features_forest, not_selected_features_forest = get_thresholds(tree, len(DATA_TABLE[0])-1, [])

    for i in range(len(CONSTANT_VALS)):
        #print(CONSTANT_VALS[i])
        #print(cutoff_dict_forest[i+1])
        CONSTANT_VALS[i].extend(cutoff_list_forest[i])
        CONSTANT_VALS[i] = sorted(list(set(CONSTANT_VALS[i])))
    return

def add_constants_from_forest(num_trees, depth):
    global CONSTANT_VALS
    dat = np.array(DATA_TABLE)
    #print(dat)
    x = pd.DataFrame(data=dat[1:,0:-1])
    #print(x)
    y = pd.DataFrame(data=dat[1:,-1])
    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.0,random_state=0)

    forest = RandomForestClassifier(max_depth=depth, n_estimators=num_trees)
    forest.fit(X_train, y_train)

    cutoff_list_forest, selected_features_forest, not_selected_features_forest, cutoff_dict_forest = get_thresholds_FOREST(forest, len(DATA_TABLE[0])-1, [])

    print(CONSTANT_VALS)
    print(cutoff_dict_forest)

    for i in range(len(CONSTANT_VALS)):
        print(CONSTANT_VALS[i])
        print(cutoff_dict_forest[i+1])
        CONSTANT_VALS[i].extend(cutoff_dict_forest[i+1])
        CONSTANT_VALS[i] = list(set(CONSTANT_VALS[i]))
    return
    
    for index in range(len(DATA_TABLE[0])-1):
        SORTED_FEATURE = index
        all_rows = sorted(list(range(len(DATA_TABLE)-1)), key=get_sorted_feature_value)
        
        cut_points = []
        previous_target = -1000
        previous_value = -1000
        old_cut_point = -1000
        for d in all_rows:
            if previous_value == get_feature_value(d,index):
                if previous_target != get_target(d):
                    put_cut
            
            if previous_target != -1000 and previous_target != get_target(d):
                cut_points.extend([float(get_feature_value(d,index)+previous_value)/2.0])
            previous_target = get_target(d)
            previous_value = get_feature_value(d,index)
        cut_points = sorted(list(set(cut_points)))
        
        CONSTANT_VALS.extend([cut_points])
    
    #print CONSTANT_VALS

# Getting tree structural values (parents, children, etc), num_nodes is the total number of nodes in the tree
#
# Example numbering for a tree of depth 3, leafs are numbered seperately:
#
#         3            nodes
#     1       5
#   0   2   4   6
#  0 1 2 3 4 5 6 7     leafs
#
#

# recursive, get total number of leafs under node
def get_num_leafs(node, num_nodes):
    check = num_nodes / 2
    if node > check:
        return get_num_leafs(node - check, check)
    if node < check:
        return get_num_leafs(node, check)
    return check

# get right leafs under node
def get_right_leafs(node, num_nodes):
    num_leafs = get_num_leafs(node+1, num_nodes+1)
    return list(range(node+1, node+1 + int(num_leafs)))

# get left leafs under node
def get_left_leafs(node, num_nodes):
    num_leafs = get_num_leafs(node+1, num_nodes+1)
    return list(range(node+1 - int(num_leafs), node+1))

# get left nodes (not leafs) under node
def get_left_nodes(node, num_nodes):
    return get_left_leafs(node, num_nodes)[0:-1]

# get right nodes (not leafs) under node
def get_right_nodes(node, num_nodes):
    return get_right_leafs(node, num_nodes)[0:-1]

def get_depth(node, num_nodes):
    #print node, num_nodes
    if node < (num_nodes-1)/2:
        return 1 + get_depth(node, ((num_nodes+1)/2) - 1)
    if node > (num_nodes-1)/2:
        return 1 + get_depth(node - ((num_nodes+1)/2), ((num_nodes+1)/2) - 1)
    return 1

def get_left_node(node, num_nodes):
    max_depth = int(math.log(num_nodes + 1, 2)) - 1
    if node % 2 == 0: return -1
    return node - (2 ** (max_depth - get_depth(node, num_nodes)))

def get_right_node(node, num_nodes):
    max_depth = int(math.log(num_nodes + 1, 2)) - 1
    if node % 2 == 0: return -1
    return node + (2 ** (max_depth - get_depth(node, num_nodes)))

# recursive, get path from node to root node
def get_path_recur(node, leaf, diff):
    if diff < 1:
        if leaf <= node:
            return [node, "left"]
        return [node, "right"]
    if leaf <= node:
        return get_path_recur(node - diff, leaf, diff / 2) + [node, "left"]
    return get_path_recur(node + diff, leaf, diff / 2) + [node, "right"]

# get path from leaf to root node
def get_path(leaf, num_nodes):
    return get_path_recur(((num_nodes+1) / 2) - 1, leaf, ((num_nodes+1) / 2) / 2)

# get path from node to root node, including node
def get_pathn(node, num_nodes):
    leaf = min(get_right_leafs(node, num_nodes))
    path = get_path(leaf, num_nodes)
    path_len = int(len(path)/2.0)
    
    for i in range(path_len):
        n = path[i*2]
        if n == node:
            return path[i*2:len(path)]
    return []

def convert_node(tree, node, num_nodes):
    path = get_pathn(node, num_nodes)
    path_len = int(len(path)/2.0)
    
    index = 0
    
    for l in reversed(list(range(1,path_len))):
        node = path[l*2]
        dir = path[l*2+1]
                
        if sget_right_node(tree, index) == -1:
            return index

        if dir == "right":
            index = sget_right_node(tree, index)
        if dir == "left":
            index = sget_left_node(tree, index)

    return index

def convert_leaf(tree, leaf, num_nodes):
    path = get_path(leaf, num_nodes)
    path_len = int(len(path)/2.0)
    
    index = 0
    
    for l in reversed(list(range(path_len))):
        node = path[l*2]
        dir = path[l*2+1]
        
        if sget_right_node(tree, index) == -1:
            return index
        
        if dir == "left":
            index = sget_right_node(tree, index)
        if dir == "right":
            index = sget_left_node(tree, index)

    return index

# recursive, used to get number of parents of node until root
def get_num_parents_recur(node, target, diff):
    if diff < 1:
        return 0
    if target == node:
        return 0
    if target > node:
        return get_num_parents_recur(node + diff, target, diff / 2) + 1
    return get_num_parents_recur(node - diff, target, diff / 2) + 1

