import pandas as pd

from corels_dt_nosimilar_multicopies import bbound


compas1 = pd.DataFrame(pd.read_csv('../data/compas-binary1.csv'))


x_compas1 = compas1.values[:,:12]
y_compas1 = compas1.values[:,12]

print("============================================All bounds===================================================")

# compas1, all 12 feature
bbound(x_compas1, y_compas1, lamb=0.005, prior_metric="curiosity", init_cart=True, incre_support=False)
#####