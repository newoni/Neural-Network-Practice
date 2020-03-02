# <20.2.29> by KH

'''
58 page
FFNN Keras implementation
'''
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard

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

#Splitting the dataset in training and test set
msk = np.random.rand(len(data)) < 0.5

# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1','x2']][msk], data[['type']][msk].values

# Everything else will go into he validation set
test_x, test_y = data[['x1','x2']][~msk], data[['type']][~msk].values

model = Sequential()
model.add(Dense(2, input_dim=2))

model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)


model.compile(loss='mse', optimizer=sgd)

model.fit(train_x[['x1', 'x2']], train_y, batch_size=1,epochs=2)

pred = model.predict_proba(test_x)
print("'MSE: ",mean_squared_error(test_y, pred))
