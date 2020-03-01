# <20.2.28> by KH
 
'''
49 page
Keras implementation
'''

from keras.layers import Activation, Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(32))
model.add(Activation('tanh'))

from keras import backend as K

model.add(Dense(32, activation=K.tanh))
