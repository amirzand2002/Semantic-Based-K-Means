# read all tweet info from files and save them in .pkl file


# imports
import os
import pandas as pd
from tqdm import tqdm

# the information needed to crawl the background tweets.
# Each .dat file corresponds to an entity and includes the same information as described for ./training/tweet_info dir


s = []
tqdm.pandas(desc="progress-bar")
data_test = pd.DataFrame()
""" read from test files
 giving the directory for the files which containing tweets information"""
for root, dirs, files in os.walk('replab2013/replab2013-dataset/test/tweet_info/'):
    for file in tqdm(files):
        filename, extension = os.path.splitext(file)
        if extension == '.dat':
            print("\n", filename)
            with open('replab2013/replab2013-dataset/test/tweet_info/%s' % file, encoding='ISO-8859-1') as f:
                data_entity_test = pd.read_table(f, encoding='ISO-8859-1')
                data_test = pd.concat([data_test, data_entity_test])
            print("\n concat %s finished\n" % filename)

    data_test.to_pickle('test_info.pkl')
    data_test.to_csv('test__info.csv')
    # save .csv and .pkl files

print("reading test tweet information finished\n")
# save .pkl files
################################
# read from train files

data_train = pd.DataFrame()
for root, dirs, files in os.walk('replab2013/replab2013-dataset/training/tweet_info/'):
    for file in tqdm(files):
        filename, extension = os.path.splitext(file)
        if extension == '.dat':
            print("\n", filename)
            with open('replab2013/replab2013-dataset/training/tweet_info/%s.dat' % filename,
                      encoding='ISO-8859-1') as f:
                data_entity_train = pd.read_table(f, encoding='ISO-8859-1')
                data_train = pd.concat([data_train, data_entity_train])
            print("\n concat %s finished\n" % filename)

    data_train.to_pickle('train_info.pkl')
    data_train.to_csv('train__info.csv')
    # save .csv and .pkl files
print("reading training tweet information finished\n")

# read test and train info .csv files and merged them into one files
test_info = pd.read_csv("test__info.csv")
train_info = pd.read_csv("train__info.csv")
frames = [test_info, train_info]
result = pd.concat(frames, ignore_index=True, sort=False)
result.to_csv('all_data_info.csv', index=False)
