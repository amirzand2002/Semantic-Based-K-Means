# Reading from tweets adn calculating jaccard similarity for each row

# imports
from typing import List, Any

import pandas as pd
import nltk
from nltk.corpus import stopwords
import preprocessor as p
import re
# from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import TweetTokenizer
import time
import requests
from tqdm import tqdm
from itertools import chain

data = pd.read_excel('tweet_words.xlsx', 'Sheet1')
data = data.drop(columns=['cleaned', 'test'])
simil_mat = pd.DataFrame()

import time
import math
from multiprocessing import Pool
from multiprocessing import freeze_support

'''Define function to run mutiple processors and pool the results together'''


def run_multiprocessing(func, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return pool.map(func, i)


def Jaccard_Similarity(doc1, doc2):
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split())
    words_doc2 = set(doc2.lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)


def calc(data, simil_mat):
    for index, row in data.iterrows():
        print(index)
        for index2, row2 in data.iterrows():
            # row == index and column == index2
            try:
                simil_mat.loc[index, index2] = Jaccard_Similarity(data.words[index], data.words[index2])
            except:
                simil_mat.loc[index, index2] = 0

    return simil_mat


def main():
    start = time.clock()

    '''
    set up parameters required by the task
    '''
    num_max = data.size
    n_processors = 6
    x_ls = list(range(num_max))

    '''
    pass the task function, followed by the parameters to processors
    '''
    out = run_multiprocessing(calc(data, simil_mat), x_ls, n_processors)

    print("Input length: {}".format(len(x_ls)))
    print("Output length: {}".format(len(out)))
    print("Mutiprocessing time: {}mins\n".format((time.clock() - start) / 60))
    print("Mutiprocessing time: {}secs\n".format((time.clock() - start)))
    simil_mat.to_excel("jaccard_similarity.xlsx")


if __name__ == "__main__":
    freeze_support()  # required to use multiprocessing
    main()
