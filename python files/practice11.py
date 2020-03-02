# <20.2.29> by KH

'''
75 page
CNN in Keras
'''

import keras
from keras.datasets import mnist    # Modified National Institute of Standards and Technology
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K

batch_size = 128
epochs = 2

# We know we have 10 classes
# which are the digits from 0~9
num_classes = 10

# the data, split between train and test sets
(X_train, y_train), (X_test,  y_test) = mnist.load_data()

# input image dimensions
img_rows, img_cols = X_train[0].shape

# Reshaping the data to use it in our network
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Scaleing the data
X_train = X_train / 255.0
X_test = X_test/255.0

plt.imshow(X_test[1][...,0], cmap='Greys')
plt.axis('off')
plt.show()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3),
                 activation = 'relu',
                 input_shape = input_shape))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

loss = 'categorical_crossentropy'
optimizer = 'adam'

model.compile(
    loss=loss, optimizer=optimizer, metrics = ['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 1,
          validation_data = (X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print(f'Test loss: { score[0]} - Test accuracy: {score[1]}')
