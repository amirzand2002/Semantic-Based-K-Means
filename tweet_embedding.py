""" import libraries """
from typing import Dict
import pickle
import pandas as pd
import numpy as np
import tqdm
from scipy import spatial
import codecs
import math
from numpy import array, ndarray
from numpy import asarray
from numpy import zeros
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

''' calculate cosine similarity'''


def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    a_norm = np.sqrt(np.sum(a ** 2))
    b_norm = np.sqrt(np.sum(b ** 2))
    denominator = a_norm * b_norm
    cosine_similarity = nominator / denominator
    return cosine_similarity


''' load the Stanford GloVe model '''
filename = 'glove.twitter.27B.25d.txt.word2vec'
# model = KeyedVectors.load_word2vec_format(filename, binary=False)

''' Reading glove and store words and their coefficient in embeddings_index '''
embeddings_index: dict[str, ndarray] = dict()
f = open('glove.twitter.27B.25d.txt')
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        f.__next__()
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

''' Read tweets from tweets_words_en '''
data = pd.read_excel("tweets_words_en.xlsx")
tweets = data.words_with_flicker

''' prepare tokenizer '''
t = Tokenizer()
t.fit_on_texts(tweets)
vocab_size = len(t.word_index) + 1

''' integer encode the documents '''
encoded_docs = t.texts_to_sequences(tweets)
# print(encoded_docs)
# create a weight matrix for words in training docs
# embedding_matrix = zeros((vocab_size, 25))
# for word, i in t.word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        embedding_matrix[i] = embedding_vector
# print(embeddings_index['iran'])
''' cosine distance = 1- cosine similarity'''
counter2 = 1
embeddings_tweet: dict[str, ndarray] = dict()
# spatial.distance.cosine(embeddings_index['iran'], embeddings_index['iraq'])
'''split each row of tweet and calculate new embedding for them with use of cosine and counter'''
for items in tweets:
    x = items.split(',')
    s = len(x)
    counter = 1

    embeddings_add = 0
    for item in x:
        try:
            # print(item)
            # print(embeddings_index[item])
            # print(math.cos(360 / s * counter))
            j = embeddings_index[item] * math.cos((360 / s) * counter)
            # print(embeddings_index[item] * math.cos((360 / s) * counter))
            embeddings_add = embeddings_add + j

        except:
            embeddings_add = embeddings_add * 1
        counter += 1
    word = items
    coefs2 = embeddings_add
    embeddings_tweet[word] = coefs2
    counter2 += 1
    print(counter2)

# save dictionary to embeddings_tweet.pkl file
with open('embeddings_tweet.pickle', 'wb') as handle:
    pickle.dump(embeddings_tweet, handle, protocol=pickle.HIGHEST_PROTOCOL)
data['embedding'] = ''
for index, row in data.iterrows():
    if data.words_with_flicker[index] in embeddings_tweet:
        data.embedding[index] = embeddings_tweet[data.words_with_flicker[index]]
data.to_excel("tweets_with_emb.xlsx")
# TODO: use data.embedding to cluster them with the usage of K-means and after that use
#  other clustering algorithm like DBSCAN
