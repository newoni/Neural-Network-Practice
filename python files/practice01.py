# <20.2.28> by KH

"""
20 page
Feature scaling
"""

from sklearn import preprocessing
import numpy as np

X_train = np.array([[-3,1,2],[2,0,0],[1,2,3]])

X_scaled = preprocessing.scale(X_train)

scaler = preprocessing.MinMaxScaler()
X_scaled2 = scaler.fit_transform(X_train)
