# <20.3.3> by KH
'''
page 96
Word embedding in Keras
'''

from numpy import array
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
import pandas as pd

# define the corpus
corpus = ['This is good pizza',
          'I love Italian pizza',
          'The best pizza',
          'nice pizza',
          'Excellent pizza',
          'I love pizza',
          'The pizza was alright',
          'disgusting pineapple pizza',
          'not good pizza',
          'bad pizza',
          'very bad pizza',
          'I had better pizza']

# creating class labels for our
labels = array([1,1,1,1,1,1,0,0,0,0,0,0])

output_dim = 8
data = pd.DataFrame({'Text':corpus, 'sentiment':labels})

# we extract the vocabulary from our corpus
sentence = [voc.split() for voc in corpus]
vocabulary = set([word for sentence in sentence for word in sentence])

vocab_size = len(vocabulary)
encoded_corpus = [one_hot(d, vocab_size) for d in corpus]

# we now pad the documents to
# the max length of the longest sentences
# to have an uniform length
max_length = 5
padded_docs = pad_sequences(encoded_corpus, maxlen=max_length, padding= 'post')

# model deinition
model = Sequential()
model.add(Embedding(vocab_size, output_dim, input_length= max_length, name='embedding'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics =['acc'])

# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose = 0)

#evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print('Accuracy:%f'%(accuracy*100))
