# read all tweet info from files and save them in .pkl file


# imports
import glob
import os
import os
import pandas as pd
import csv
from tqdm import tqdm


# the information needed to crawl the background tweets.
# Each .dat file corresponds to an entity and includes the same information as described for ./training/tweet_info dir


path = '/replab2013-dataset/background/tweet_info/*.dat'
s = []

tqdm.pandas(desc="progress-bar")
de = pd.DataFrame()
# read from test files
for root, dirs, files in os.walk('replab2013-dataset/test/tweet_info/'):
    for file in tqdm(files):
        filename, extension = os.path.splitext(file)
        if extension == '.dat':
            print(filename)
            with open('replab2013-dataset/test/tweet_info/%s.dat' % filename, encoding='ISO-8859-1') as f:
                de1 = pd.read_table(f, encoding='ISO-8859-1')
                de = pd.concat([de, de1])
            print("concate %s finished" % filename)

    de.to_pickle('test_info.pkl')
    de.to_csv('test__info.csv')
# save .pkl files
################################
# read from train files

de = pd.DataFrame()
for root, dirs, files in os.walk('replab2013-dataset/training/tweet_info/'):
    for file in tqdm(files):
        filename, extension = os.path.splitext(file)
        if extension == '.dat':
            print(filename)
            with open('replab2013-dataset/training/tweet_info/%s.dat' % filename, encoding='ISO-8859-1') as f:
                de1 = pd.read_table(f, encoding='ISO-8859-1')
                de = pd.concat([de, de1])
            print("concate %s finished" % filename)

    de.to_pickle('train_info.pkl')
    de.to_csv('train__info.csv')
# save .pkl files
