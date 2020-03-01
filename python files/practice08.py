# <20.2.28> by KH

'''
52 page
The XOR problem
'''
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# matplotlb.use("TKAgg")

# initiating random number
np.random.seed(11)

#### Ceating the dataset

# mean and standard deviation for the x belonging to the first clas
mu_x1, sigma_x1 = 0, 0.1

# Constant to make the second distribution different from the first
x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5
# x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0, 1, 0, 1

# creating the first distribution
d1 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
                   'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
                   'type':0})

d2 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x2_mu_diff,
                   'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
                   'type':1})

d3 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
                   'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
                   'type':0})

d4 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
                   'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x4_mu_diff,
                   'type':1})

data = pd.concat([d1, d2, d3, d4], ignore_index=True)

plt.figure()
plt.scatter(x=data[data.columns[0]], y=data[data.columns[1]], alpha=0.2,c=data[data.columns[2]])
plt.colorbar()
