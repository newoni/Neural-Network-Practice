# <20.3.1>

'''
78 page
Network configuraion
'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import itertools
import os

batch_size = 512
num_classes = 10
epochs = 1
N_SAMPLES = 30000

model_directory = 'models'

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# input image dimensions
img_rows, img_cols = X_train[0].shape

# Reshaping the data to use it in our network
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

# Scaling the data
X_train = X_train / 255.0
X_test = X_test/255.0

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

loss = 'categorical_crossentropy'
optimizer = 'adam'

X_train = X_train[:N_SAMPLES]
X_test = X_test[:N_SAMPLES]

y_train = y_train[:N_SAMPLES]
y_test = y_test[:N_SAMPLES]

filters = [4,8,16]
kernal_sizes = [(2,2), (4,4), (16,16)]

config = itertools.product(filters, kernal_sizes)

for n_filters, kernel_size in config:
    model_name = 'single_f_' + str(n_filters) + '_k_' + str(kernel_size)

    model = Sequential(name = model_name)
    model.add(
        Conv2D(n_filters, kernel_size = kernel_size, activation = 'relu', input_shape = input_shape))
    model.add(Flatten())
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(loss= loss, optimizer =optimizer, metrics=['accuracy'])

    model.fit(
        X_train,
        y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        validation_data = (X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)

    print(model_name, 'Test loss:', score[0], 'Test accuracy:', score[1])

    os.system("mkdir " + model_directory)
    model_path = os.path.join(model_directory,model_name)
    model.save(model_path)
