# <20.04.26> by KH

'''
107 page ~ 108 page
Text classification with GloVe

embeddings_index가 정의되지 않았으므로 실행X
'''
import os

possible_word_vectors = (50, 100, 200, 300)
word_vectors = possible_word_vectors[0]
file_name = f'glove.6B.{word_vectors}d.txt'
filepath= '../data/'
pretrained_embedding = os.path.join(filepath, file_name)

class EmbeddingVectorizer(object):
    '''
    Follows the scikit-learn API
    Transform each document in the average
    of the embedding of the words in it
    '''

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim=50

    def fit(self, X, y):
        return self

    def transform(self, X):
        '''
        Find the embedding vector for each word in the dictionary
        and take the mean for each document
        '''
        # Renaming it just to make it more understandable
        documents = X
        embedded_docs = []
        for words in documents:
            # For each document
            # Consider the mean of the embeddings
            embedded_document = []
            for words in document:
                for w in words:
                    if w in self.word2vec:
                        embedded_word = self.word2vec[w]
                    else:
                        embedded_word = np.zeros(self.dim)
                    embedded_document.append(embedded_word)
            embedded_docs.append(embedded_document, axis = 0)
        return embedded_docs

# Creating the embedding
e = EmbeddingVectorizer(embeddings_index)
X_train_embedded = e.transform(X_train)

# Train the classifier
rf = RandomForestClassifier(n_estimators=50, n_jobs = 1)
rf.fit(X_train_embedded, y_train)
X_train_embedded = e.transform(X_test)
predictions = rf.predict(X_test_embedded)

print('AUC score: ', roc_auc_scroe(predictions, y_test))
comfusion_matrix(predictions, y_test)
