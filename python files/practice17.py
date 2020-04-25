# <20.04.26> by KH

'''
104 page ~ 106 page
Using the GloVe model

glove model 사용.
import 문제로 사용못함.
'''

import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove

sentences = list(itertools.islice(Text8Corpus('text8'),None))

corpus = Corpus()

corpus.fit(sentences, window = 10)
glove = Glove(no_components= 100, learning_rate = 0.05)

glove.fit(corpus.matrix, epochs =30, no_threads=4, verbose = True)

glove.add_dictionary(corpus.dictionary)

glove.most_similar('man')
glove.most_similar('man', number=10)
glove.most_similar('frog', number=10)
glove.most_similar('girl', numbver=10)
glove.most_similar('car', numbver=10)
glove.most_similar('queen', numbver=10)