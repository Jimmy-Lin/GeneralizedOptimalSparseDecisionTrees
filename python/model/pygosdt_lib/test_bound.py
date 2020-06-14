import pandas as pd

from corels_dt_nosimilar_multicopies import bbound


compas1 = pd.DataFrame(pd.read_csv('../data/compas-binary1.csv'))


x_compas1 = compas1.values[:,:12]
y_compas1 = compas1.values[:,12]


# different priority queue


print("============================================PriorQueue by curiosity===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity")
#####

print("============================================PriorQueue by objective===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="objective")
#####

print("============================================PriorQueue by lower bound===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="bound")
#####

print("============================================PriorQueue by entropy===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="entropy")
#####

print("============================================PriorQueue by gini===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="gini")
#####

"""

print("============================================All bounds===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity")
#####

print("============================================No support===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity", support=False)
#####

print("============================================No accu_support===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity", accu_support=False)
#####

print("============================================No incre_support===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity", incre_support=False)
#####

print("============================================No lookahead===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity", lookahead=False)
#####

print("============================================No equiv_points===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity", equiv_points=False)
#####
"""

