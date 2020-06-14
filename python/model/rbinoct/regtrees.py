import random, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import tree, ensemble, linear_model, svm #, cross_validation
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


def get_data(inputfile):
    global df
    df = pd.read_csv(inputfile, sep=';')
    return df

def learnTrees(depth):
    global dt
    global features
    global targets
    features = list(df.columns)
    target_feature = features[-1]
    features = list(features[:len(features)-1])
    targets = df[target_feature].unique()
    print('targets:', targets)
    print('features:', features)
    y=df[target_feature]
    X=df[features]
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.0,random_state=0)
    X_train = X_test = X
    y_train = y_test = y
    #dt = DecisionTreeRegressor(max_depth=depth)#, min_samples_split=20, random_state=99)
    dt = DecisionTreeClassifier(max_depth=depth)#, min_samples_split=20, random_state=99)
    dt.fit(X_train,y_train)
    prediction = dt.predict(X_train)
    print('accuracy:', accuracy_score(y_train,prediction))
    print('num correct:', accuracy_score(y_train,prediction) * len(y_train))
    print('R2 Score:', r2_score(y_train,prediction))
    print('absolute error:', mean_absolute_error(y_train,prediction)*len(X_train))

def learnRegTrees(depth):
    global dt
    global features
    global targets
    features = list(df.columns)
    target_feature = features[-1]
    features = list(features[:len(features)-1])
    targets = df[target_feature].unique()
    print('targets:', targets)
    print('features:', features)
    y=df[target_feature]
    X=df[features]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.0,random_state=0)
    dt = DecisionTreeRegressor(max_depth=depth, criterion="mae")#, min_samples_split=20, random_state=99)
    #dt = DecisionTreeClassifier(max_depth=depth)#, min_samples_split=20, random_state=99)
    dt.fit(X_train,y_train)
    prediction = dt.predict(X_train)
    #print 'accuracy:', accuracy_score(y_train,prediction)
    #print 'num correct:', accuracy_score(y_train,prediction) * len(y_train)
    print('R2 Score:', r2_score(y_train,prediction))
    print('absolute error:', mean_absolute_error(y_train,prediction)*len(X_train))

def get_code(spacer_base="    "):#tree, feature_names, target_names, spacer_base="    "):
    """Produce psuedo-code for decision tree.
        
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

    return None

    tree = dt
    feature_names = features
    target_names = targets
    
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    print(">>>>>>>>>>tree.tree_.feature>>>>>>>>>", tree.tree_.feature)
    print(">>>>>>>>>>tree.tree_>>>>>>>>>", tree.tree_)
    print(">>>>>>>>>>feature_names>>>>>>>>>", feature_names)
    feats  = ['X']#[feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    
    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print((spacer + "if ( " + feats[node] + " <= " + \
                  str(threshold[node]) + " ) {"))
            if left[node] != -1:
                recurse(left, right, threshold, feats,left[node], depth+1)
            print((spacer + "}\n" + spacer +"else {"))
            if right[node] != -1:
                recurse(left, right, threshold, feats, right[node], depth+1)
            print((spacer + "}"))
        else:
            target = value[node]
            print((spacer + "return " + str(target)))
            for i, v in zip(np.nonzero(target)[1], target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print((spacer + "return " + str(target_name) + " " + str(i) + " "\
                      " ( " + str(target_count) + " examples )"))
                      

    recurse(left, right, threshold, features, 0, 0)

"""
def visulize_tree(tree):
    with open("wine.dot",'w') as f:
        f = tree.export_graphviz(dt, out_file=f)
"""

def run_tree():
    df = get_wine_data()
    learnTrees(3)
    get_code()#dt, features, targets)

#run_tree()