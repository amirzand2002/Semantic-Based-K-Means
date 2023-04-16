""" import libraries """
from sklearn.cluster import KMeans
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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

''' Read tweets from tweets_words_en '''
data = pd.read_excel("tweets_with_emb.xlsx")

# Get embeddings
embeddings = np.array([np.fromstring(embedding.strip('[]'), dtype=float, sep=' ') for embedding in data['embedding']])

# Compute cosine similarity matrix
cos_sim = cosine_similarity(embeddings)

# Cluster using k-means
kmeans = KMeans(n_clusters=20, random_state=0, max_iter=300, n_init=10, algorithm='auto')
kmeans.fit(cos_sim)

# Get cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to data
data['label'] = cluster_labels

# Save data with cluster labels
data.to_excel("tweets_with_emb_clustered.xlsx", index=False)
# TODO: kmeans clustering is done based on their cosine similarity of tweets' embedding
# TODO: implement DBSCAN Algorithm on Data
# TODO: Check the Clustering result
