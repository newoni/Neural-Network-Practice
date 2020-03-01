# <20.2.28>

"""
20 page
Feature scaling
"""
# from sklearn import preprocessing
# import numpy as np
#
# X_train = np.array([[-3,1,2],[2,0,0],[1,2,3]])
#
# X_scaled = preprocessing.scale(X_train)
#
# scaler = preprocessing.MinMaxScaler()
# X_scaled2 = scaler.fit_transform(X_train)


"""
21 page
keras feature engineering
"""
# from keras.preprocessing.image import ImageDataGenerator
#
# datagen = ImageDataGenerator(rotation_range = 45,           # 회전 (0~180)
#                              width_shift_range=0.25,        #
#                              height_shift_range = 0.25,
#                              rescale = 1./255,
#                              shear_range = 0.3,             # shearing 해줌 -> 기울어짐
#                              zoom_range = 0.3,
#                              horizontal_flip = True,
#                              fill_mode = 'nearest')

'''
23 page
Supervised learning algorithms
'''

# from sklearn.datasets.california_housing import fetch_california_housing
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# # Using a standard dataset that we ca find in scikit-learn
# cal_house = fetch_california_housing()
#
# # Split the input data into training/testing sets
# cal_house_X_train = cal_house.data[:-20]
# cal_house_X_test = cal_house.data[-20:]
#
# # Split the targets into training/testing sets
# cal_house_Y_train = cal_house.target[:-20]
# cal_house_Y_test = cal_house.target[-20:]
#
# # Created linear regression object
# regr = LinearRegression()
#
# # Train the model using the training sets
# regr.fit(cal_house_X_train, cal_house_Y_train)
#
# # Calculating he predictions
# predictions = regr.predict(cal_house_X_test)
#
# #Calculating the loss
# print('MSE: {:.2f}'.format(mean_squared_error(cal_house_Y_test,predictions)))

'''
25 page
Evaluatng the model
'''

# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from sklearn.linear_model import LinearRegression
# from sklearn import datasets
#
# #import some data
# iris = datasets.load_iris()
#
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state = 0)
#
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_transformed = scaler.transform(X_train)
# X_test_transformed = scaler.transform(X_test)
#
# clf = LinearRegression().fit(X_train_transformed, y_train)
#
# predictions = clf.predict(X_test_transformed)
#
# print('Predictions:', predictions)

'''
32 page
Implementing a perceptron
'''

# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
#
# # initiating random number
# np.random.seed(11)
#
# # mean and standard deviation for the x belonging to the first class
# mu_x1, sigma_x1 = 0, 0.1
#
# # constat to make the second distribution different from the first
# x2_mu_diff = 0.35
#
# # creating the first distribution
# d1 = pd.DataFrame({'x1':np.random.normal(mu_x1,sigma_x1,1000),'x2':np.random.normal(mu_x1,sigma_x1,1000),'type':0})
#
# # creating the second distribution
# d2 = pd.DataFrame({'x1':np.random.normal(mu_x1,sigma_x1,1000) + x2_mu_diff,'x2':np.random.normal(mu_x1,sigma_x1,1000) + x2_mu_diff,'type':1})
#
# data = pd.concat([d1,d2], ignore_index = True)
#
# ax = sns.scatterplot(x="x1",y='x2',hue = 'type',data=data)
#
# class Perceptron:
#     # Simpole implementation of the perceptron algorithm
#
#     def __init__(self,w0=1,w1=0.1,w2=0.1):
#         # weights
#
#         self.w0 = w0 # bias
#         self.w1 = w1
#         self.w2 = w2
#
#     def step_function(self,z):
#         if z>=0:
#             return 1
#         else:
#             return 0
#
#     def weighed_sum_inputs(self,x1,x2):
#         return sum([1*self.w0, x1*self.w1, x2*self.w2])
#
#     def predict(self, x1, x2):
#         # Uses the step function to determine the output
#
#         z = self.weighed_sum_inputs(x1,x2)
#         return self.step_function(z)
#
#     def predict_boundary(self, x):
#         # used to predict the boundaries of our classifier
#
#         return -(self.w1 * x + self.w0)/self.w2
#
#     def fit(self,X,y, epochs = 1, step = 0.1, verbose = True):
#         # Train the model given the dataset
#
#         errors = []
#
#         for epoch in range(epochs):
#             error = 0
#             for i in range(0,len(X.index)):
#                 x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
#
#                 # The update is proportional to the step size and the error
#
#                 update = step * (target - self.predict(x1, x2))
#
#                 self.w1 += update * x1
#                 self.w2 += update * x2
#                 self.w0 += update
#                 error += int(update != 0.0)
#
#             errors.append(error)
#
#             if verbose:
#                 print('Epochs: {} - Error: {} - Errors from all epochs: {}'.format(epoch,error,errors))
#
# # Splitting the dataset in training and test set
# msk = np.random.rand(len(data))< 0.8
#
# # Roughly 80% of data will go in the training set
# train_x, train_y = data[['x1','x2']][msk], data.type[msk]
#
# #Everything else will go into the valitation set
# test_x, test_y = data[['x1','x2']][~msk], data.type[~msk]
#
# my_perceptron = Perceptron(0.1, 0.1)
# my_perceptron.fit(train_x, train_y, epochs= 1, step=0.005)
#
# pred_y = test_x.apply(lambda x: my_perceptron.predict(x.x1, x.x2), axis=1)
#
# cm = confusion_matrix(test_y, pred_y, labels = [0,1])
# print(pd.DataFrame(cm, index = ['True 0', 'True 1'],
#                    columns = ['Predicted 0', 'Predicted 1']))
#
# # Adds decision boundary line to the sctterplot
# ax = sns.scatterplot(x='x1', y='x2', hue='type', data=data[~msk])
#
# ax.autoscale(False)
#
# x_vals = np.array(ax.get_xlim())
# y_vals = my_perceptron.predict_boundary(x_vals)
# ax.plot(x_vals, y_vals, '--', c='red')

'''
39 page
Implementing perceptron in Keras
'''
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
#
# import keras
#
# # initiating random number
# np.random.seed(11)
#
# # mean and standard deviation for the x belonging to the first class
# mu_x1, sigma_x1 = 0, 0.1
#
# # constat to make the second distribution different from the first
# x2_mu_diff = 0.35
#
# # creating the first distribution
# d1 = pd.DataFrame({'x1':np.random.normal(mu_x1,sigma_x1,1000),'x2':np.random.normal(mu_x1,sigma_x1,1000),'type':0})
#
# # creating the second distribution
# d2 = pd.DataFrame({'x1':np.random.normal(mu_x1,sigma_x1,1000) + x2_mu_diff,'x2':np.random.normal(mu_x1,sigma_x1,1000) + x2_mu_diff,'type':1})
#
# data = pd.concat([d1,d2], ignore_index = True)
#
# ax = sns.scatterplot(x="x1",y='x2',hue = 'type',data=data)
#
# class Perceptron:
#     # Simpole implementation of the perceptron algorithm
#
#     def __init__(self,w0=1,w1=0.1,w2=0.1):
#         # weights
#
#         self.w0 = w0 # bias
#         self.w1 = w1
#         self.w2 = w2
#
#     def step_function(self,z):
#         if z>=0:
#             return 1
#         else:
#             return 0
#
#     def weighed_sum_inputs(self,x1,x2):
#         return sum([1*self.w0, x1*self.w1, x2*self.w2])
#
#     def predict(self, x1, x2):
#         # Uses the step function to determine the output
#
#         z = self.weighed_sum_inputs(x1,x2)
#         return self.step_function(z)
#
#     def predict_boundary(self, x):
#         # used to predict the boundaries of our classifier
#
#         return -(self.w1 * x + self.w0)/self.w2
#
#     def fit(self,X,y, epochs = 1, step = 0.1, verbose = True):
#         # Train the model given the dataset
#
#         errors = []
#
#         for epoch in range(epochs):
#             error = 0
#             for i in range(0,len(X.index)):
#                 x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
#
#                 # The update is proportional to the step size and the error
#
#                 update = step * (target - self.predict(x1, x2))
#
#                 self.w1 += update * x1
#                 self.w2 += update * x2
#                 self.w0 += update
#                 error += int(update != 0.0)
#
#             errors.append(error)
#
#             if verbose:
#                 print('Epochs: {} - Error: {} - Errors from all epochs: {}'.format(epoch,error,errors))
#
# # Splitting the dataset in training and test set
# msk = np.random.rand(len(data))< 0.8
#
# # Roughly 80% of data will go in the training set
# train_x, train_y = data[['x1','x2']][msk], data.type[msk]
#
# #Everything else will go into the valitation set
# test_x, test_y = data[['x1','x2']][~msk], data.type[~msk]
#
# my_perceptron= keras.models.Sequential()
#
# input_layer = keras.layers.Dense(1, input_dim= 2, activation="sigmoid", kernel_initializer='zero')
# my_perceptron.add(input_layer)
#
# my_perceptron.compile(loss = 'mse', optimizer = keras.optimizers.SGD(lr=0.01))   # SGD = Stochastic Gradient Descent
# my_perceptron.fit(train_x.values, train_y, nb_epoch =2 , batch_size = 32, shuffle=True)
#
# from sklearn.metrics import roc_auc_score
#
# pred_y = my_perceptron.predict(test_x)
# print(roc_auc_score(test_y,pred_y))

'''
49 page
Keras implementation
'''

# from keras.layers import Activation, Dense
# from keras.models import Sequential
#
# model = Sequential()
# model.add(Dense(32))
# model.add(Activation('tanh'))
#
# from keras import backend as K
#
# model.add(Dense(32, activation=K.tanh))

'''
52 page
The XOR problem
'''
# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
#
# # matplotlb.use("TKAgg")
#
# # initiating random number
# np.random.seed(11)
#
# #### Ceating the dataset
#
# # mean and standard deviation for the x belonging to the first clas
# mu_x1, sigma_x1 = 0, 0.1
#
# # Constant to make the second distribution different from the first
# x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5
# # x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0, 1, 0, 1
#
# # creating the first distribution
# d1 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'type':0})
#
# d2 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x2_mu_diff,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'type':1})
#
# d3 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
#                    'type':0})
#
# d4 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x4_mu_diff,
#                    'type':1})
#
# data = pd.concat([d1, d2, d3, d4], ignore_index=True)
#
# plt.figure()
# plt.scatter(x=data[data.columns[0]], y=data[data.columns[1]], alpha=0.2,c=data[data.columns[2]])
# plt.colorbar()


# <20.2.29>
'''
54 page
FFNN in python from scratch
'''
# import numpy as np
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
#
# # matplotlb.use("TKAgg")
#
# # initiating random number
# np.random.seed(11)
#
# #### Ceating the dataset
#
# # mean and standard deviation for the x belonging to the first clas
# mu_x1, sigma_x1 = 0, 0.1
#
# # Constant to make the second distribution different from the first
# x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5
# # x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0, 1, 0, 1
#
# # creating the first distribution
# d1 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'type':0})
#
# d2 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x2_mu_diff,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'type':1})
#
# d3 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
#                    'type':0})
#
# d4 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
#                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x4_mu_diff,
#                    'type':1})
#
# data = pd.concat([d1, d2, d3, d4], ignore_index=True)
#
# plt.figure()
# plt.scatter(x=data[data.columns[0]], y=data[data.columns[1]], alpha=0.2,c=data[data.columns[2]])
# plt.colorbar()
#
# class FFNN:
#     def __init__(self, input_size=2, hidden_size=2, output_size=1):
#         # Adding 1 as it will be our bias
#         self.input_size = input_size + 1
#         self.hidden_size = hidden_size + 1
#         self.output_size = output_size + 1
#
#         self.o_error = 0
#         self.o_delta = 0
#         self.z1 = 0
#         self.z1 = 0
#         self.z3 = 0
#         self.z2_error = 0
#
#         # The whole weight matrix, from the inputs till the hidden layer
#         self.w1  = np.random.randn(self.input_size, self.hidden_size)
#
#         # The final set of weights from the hidden  layer till the output layer
#         self.w2 = np.random.randn(self.hidden_size,self.output_size)    # np.random.rand 는 난수, randn은 가우시한 정규분포
#
#     def sigmoid(self, s):
#         # Activation function
#         return 1/( 1 + np.exp(-s))
#
#     def sigmoid_prime(self,s):
#         # Derivative of the sigmoid
#         return self.sigmoid(s) * (1-self.sigmoid(s))
#
#     def forward(self,X):
#         # Forward Propagation through our networkd
#         X['bias'] = 1   # Adding 1 to the inputs to include the bias in the weight
#
#         self.z1 = np.dot(X, self.w1)    # dot product of X (input) and first set of 3x2 weights
#         self.z2 = self.sigmoid(self.z1)      # dot product of hidden layer (z2) and second set of 3x1 weights
#         self.z3= np.dot(self.z2, self.w2)   # dot product of hidden layer (z2) and second set of 3x1 weights
#         o = self.sigmoid(self.z3)            # final zctivation function
#         return o
#
#     def predict(self,X):
#         return self.forward(X)
#
#     def backward(self, X, y, output, step):
#         # Backward propagation of the errors
#         X['bias'] = 1   # Adding 1 to the inputs to include  the bias in the weight
#
#         self.o_error = y - output   # errors in output
#         self.o_delta = self.o_error * self.sigmoid_prime(output) * step  # applying derivative of sigmoid to error
#
#         self.z2_error = self.o_delta.dot(self.w2.T) # z2 error: how much our hidden layer weights contributed to output error
#         self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2) * step  # applying derivative of sigmoid to z2 error
#
#         self.w1 += X.T.dot(self.z2_delta)  # adjusting first of weights
#         self.w2 += self.z2.T.dot(self.o_delta) # adjuting second set of weights
#
#     def fit(self, X, y, epochs=10, step=0.05):
#         for epoch in range(epochs):
#             X['bias']= 1    # Adding 1 to the inputs to include the bias in the weight
#
#             output = self.forward(X)
#             self.backward(X, y, output, step)
#
# #Splitting the dataset in training and test set
# msk = np.random.rand(len(data)) < 0.5
#
# # Roughly 80% of data will go in the training set
# train_x, train_y = data[['x1','x2']][msk], data[['type']][msk].values
#
# # Everything else will go into he validation set
# test_x, test_y = data[['x1','x2']][~msk], data[['type']][~msk].values
#
#
# my_network = FFNN()
# my_network.fit(train_x, train_y, epochs=1000, step=0.001)
#
# pred_y = test_x.apply(my_network.forward, axis=1)
#
# # Reshaping the data
# test_y_ = [i[0] for i in test_y]
# pred_y_ = [i[0] for i in pred_y]
#
# print('MSE: ', mean_squared_error(test_y_,pred_y_))
# print('AUC: ', roc_auc_score(test_y_, pred_y_))
#
# threshold = 0.5
# pred_y_binary = [0 if i> threshold else 1 for i in pred_y_]
#
# cm = confusion_matrix(test_y_, pred_y_binary, labels = [0,1])
#
# print(pd.DataFrame(cm, index =['True 0', 'True 1'], columns = ['Predicted 0', 'Predicted 1']))

'''
58 page
FFNN Keras implementation
'''
# import numpy as np
# # import pandas as pd
# # from keras.models import Sequential
# # from keras.layers import Dense, Dropout, Activation
# # from keras.optimizers import SGD
# # from sklearn.metrics import mean_squared_error
# # import os
# # import matplotlib.pyplot as plt
# # from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, TensorBoard
# #
# # # initiating random number
# # np.random.seed(11)
# #
# # #### Ceating the dataset
# #
# # # mean and standard deviation for the x belonging to the first clas
# # mu_x1, sigma_x1 = 0, 0.1
# #
# # # Constant to make the second distribution different from the first
# # x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5
# # # x1_mu_diff, x2_mu_diff,x3_mu_diff, x4_mu_diff = 0, 1, 0, 1
# #
# # # creating the first distribution
# # d1 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
# #                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
# #                    'type':0})
# #
# # d2 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x2_mu_diff,
# #                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+0,
# #                    'type':1})
# #
# # d3 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
# #                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x3_mu_diff,
# #                    'type':0})
# #
# # d4 = pd.DataFrame({'x1':np.random.normal(mu_x1, sigma_x1, 1000)+0,
# #                    'x2':np.random.normal(mu_x1, sigma_x1, 1000)+x4_mu_diff,
# #                    'type':1})
# #
# # data = pd.concat([d1, d2, d3, d4], ignore_index=True)
# #
# # plt.figure()
# # plt.scatter(x=data[data.columns[0]], y=data[data.columns[1]], alpha=0.2,c=data[data.columns[2]])
# # plt.colorbar()
# #
# # #Splitting the dataset in training and test set
# # msk = np.random.rand(len(data)) < 0.5
# #
# # # Roughly 80% of data will go in the training set
# # train_x, train_y = data[['x1','x2']][msk], data[['type']][msk].values
# #
# # # Everything else will go into he validation set
# # test_x, test_y = data[['x1','x2']][~msk], data[['type']][~msk].values
# #
# # model = Sequential()
# # model.add(Dense(2, input_dim=2))
# #
# # model.add(Activation('tanh'))
# # model.add(Dense(1))
# # model.add(Activation('sigmoid'))
# #
# # sgd = SGD(lr=0.1)
# #
# #
# # model.compile(loss='mse', optimizer=sgd)
# #
# # model.fit(train_x[['x1', 'x2']], train_y, batch_size=1,epochs=2)
# #
# # pred = model.predict_proba(test_x)
# # print("'MSE: ",mean_squared_error(test_y, pred))

'''
75 page
CNN in Keras
'''

# import keras
# from keras.datasets import mnist    # Modified National Institute of Standards and Technology
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# import numpy as np
# from matplotlib import pyplot as plt
# from keras import backend as K
#
# batch_size = 128
# epochs = 2
#
# # We know we have 10 classes
# # which are the digits from 0~9
# num_classes = 10
#
# # the data, split between train and test sets
# (X_train, y_train), (X_test,  y_test) = mnist.load_data()
#
# # input image dimensions
# img_rows, img_cols = X_train[0].shape
#
# # Reshaping the data to use it in our network
# if K.image_data_format() == 'channels_first':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# # Scaleing the data
# X_train = X_train / 255.0
# X_test = X_test/255.0
#
# plt.imshow(X_test[1][...,0], cmap='Greys')
# plt.axis('off')
# plt.show()
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size = (3,3),
#                  activation = 'relu',
#                  input_shape = input_shape))
# model.add(Conv2D(32, (3, 3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(num_classes, activation='softmax'))
#
# loss = 'categorical_crossentropy'
# optimizer = 'adam'
#
# model.compile(
#     loss=loss, optimizer=optimizer, metrics = ['accuracy'])
#
# model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose = 1,
#           validation_data = (X_test, y_test))
# score = model.evaluate(X_test, y_test, verbose=0)
#
# print(f'Test loss: { score[0]} - Test accuracy: {score[1]}')

'''
78 page
Network configuraion
'''
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# import itertools
# import os
#
# batch_size = 512
# num_classes = 10
# epochs = 1
# N_SAMPLES = 30000
#
# model_directory = 'models'
#
# # the data, split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # input image dimensions
# img_rows, img_cols = X_train[0].shape
#
# # Reshaping the data to use it in our network
# X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#
# input_shape = (img_rows, img_cols, 1)
#
# # Scaling the data
# X_train = X_train / 255.0
# X_test = X_test/255.0
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test,num_classes)
#
# loss = 'categorical_crossentropy'
# optimizer = 'adam'
#
# X_train = X_train[:N_SAMPLES]
# X_test = X_test[:N_SAMPLES]
#
# y_train = y_train[:N_SAMPLES]
# y_test = y_test[:N_SAMPLES]
#
# filters = [4,8,16]
# kernal_sizes = [(2,2), (4,4), (16,16)]
#
# config = itertools.product(filters, kernal_sizes)
#
# for n_filters, kernel_size in config:
#     model_name = 'single_f_' + str(n_filters) + '_k_' + str(kernel_size)
#
#     model = Sequential(name = model_name)
#     model.add(
#         Conv2D(n_filters, kernel_size = kernel_size, activation = 'relu', input_shape = input_shape))
#     model.add(Flatten())
#     model.add(Dense(num_classes, activation = 'softmax'))
#
#     model.compile(loss= loss, optimizer =optimizer, metrics=['accuracy'])
#
#     model.fit(
#         X_train,
#         y_train,
#         batch_size = batch_size,
#         epochs = epochs,
#         verbose = 1,
#         validation_data = (X_test, y_test))
#     score = model.evaluate(X_test, y_test, verbose=0)
#
#     print(model_name, 'Test loss:', score[0], 'Test accuracy:', score[1])
#
#     os.system("mkdir " + model_directory)
#     model_path = os.path.join(model_directory,model_name)
#     model.save(model_path)

'''
80 page
Keras for expression recognition
파일이 없어서 안 돌려봄.
'''
#
# import os
# import pandas as pd
# from PIL import Image
#
# # Pixel values range from 0 to 255 ( 0 is normally black and 255 is white)
# os.system("mkdir data")
# os.system("cd data")
# os.system("mkdir raw")
# basedir = os.path.join('data','raw')
# file_origin = os.path.join(basedir, 'fer2013.csv')
# data_raw = pd.read_csv(file_origin)
#
# data_input = pd.DataFrame(data_raw, columns = ['emotions', 'pixels', 'Usage'])
#
# data_input.rename({'Usage':'usage'}, inplace = True)
# data_input.head()
#
# label_map = {
#     0: '0_Anger',
#     1: '1_Disgust',
#     2: '2_Fear',\
#     3: '3_Happy',
#     4: '4_Neutral',
#     5: '5_Sad',\
#     6: '6_Surprise'
# }
#
# # Creating the folders
# output_folers = data_input['Usage'].unique().tolist()
# all_folders = []
#
# for folder in output_folders:
#     for label in label_map:
#         all_folders.append(os.path.join(basedir, folder, label_map[label]))
#
# for folder in all_folders:
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#
#     else:
#         print('Folder {} exists already'.format(folder))
#
# counter_error = 0
# counter_correct = 0
#
# def save_image(np_array_flat, file_name):
#     try:
#         im = Image.fromarray(np_array_flat)
#         im.save(file_name)
#
#     except AttributeError as e:
#         print(e)
#         return
#
# for folder in all_folders:
#     emotion = foler.split('/')[-1]
#     usage = folder.split('/')[-2]
#
#     for key, value in label_map.items():
#         if value == emotion:
#             emotion_id = key
#
#     df_to_save = data_input.reset_index()[data_input.Usage == usage][data_input.emotion == emotion_id]
#     print('saving in: ', folder, ' size: ', df_to_save.shape)
#     df_to_save['image'] = df_to_save.pixels.apply(to_image)
#     df_to_save['file_name'] = folder + '/image_' + df_to_save.index.map(str) + '_' + df_to_save.emotion.apply(str) + '-' + df_to_save.emotion.apply(lambda x: label_map[x]) + '.png'
#     df_to_save[['image', 'file_name']].apply(lambda x: save_image(x.image, x.file_name), axis=1)
#     df_to_save.apply(lambda x: save_image(x.pixels, os.path.join(basedir,x.file_name)), axis=1)
#