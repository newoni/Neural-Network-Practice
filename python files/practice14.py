# <20.3.3> by KH

'''
page 91
Machine learning for NLP
'''

import nltk
from nltk.stem.porter import *


ps = PorterStemmer()

print(ps.stem('run'))
print(ps.stem('dogs'))
print(ps.stem('mice'))
print(ps.stem('were'))

sentence = "hello, my name is KH. Machine learning is funny!"
words = nltk.word_tokenize(sentence)

print(words)

print([ps.stem(w) for w in words] )
