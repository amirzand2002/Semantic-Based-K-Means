# imports

import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import gensim.downloader as api


# read word2vec vectors of tweets from tweets-vector-2.pkl
def read_tweets(tweet_file):
    # read entity id from string
    # drop all extra entity which doesn't have relation to music
    # create entity name which is label of data
    tweet_data_frame = pd.DataFrame(pd.read_pickle(tweet_file), columns=['url', 'text',
                                                                         'id', 'author',
                                                                         'entity_id', 'tweet_url',
                                                                         'language', 'timestamp',
                                                                         'urls',
                                                                         'extended_urls',
                                                                         'md5_extended_urls',
                                                                         'is_near_duplicate_of',
                                                                         'set', 'tokenized', 'text_vec', 'set_vec'])
    # drop  unnecessary column
    tweet_data_frame = tweet_data_frame.drop(
        columns=['tweet_url', 'timestamp', 'urls', 'extended_urls', 'md5_extended_urls',
                 'is_near_duplicate_of', 'tokenized'])
    tweet_data_frame['wrd_set'] = ''
    tweet_data_frame.wrd_set = tweet_data_frame.text + ' ' + tweet_data_frame.set

    tweet_data_frame = tweet_data_frame[
        tweet_data_frame.entity_id.isin(['RL2013D04E145', 'RL2013D04E146', 'RL2013D04E149', 'RL2013D04E151',
                                         'RL2013D04E152', 'RL2013D04E153',
                                         'RL2013D04E155', 'RL2013D04E159', 'RL2013D04E161', 'RL2013D04E162',
                                         'RL2013D04E164',
                                         'RL2013D04E166', 'RL2013D04E167', 'RL2013D04E169', 'RL2013D04E175',
                                         'RL2013D04E185',
                                         'RL2013D04E194', 'RL2013D04E198', 'RL2013D04E206', 'RL2013D04E207'])]
    tweet_data_frame['entity_name'] = ''
    tweet_data_frame['entity_name_id'] = ''

    for index, row in tqdm(tweet_data_frame.iterrows()):
        if row.entity_id == 'RL2013D04E145':
            row.entity_name = 'Adele'
            row.entity_name_id = '1'
        if row.entity_id == 'RL2013D04E146':
            row.entity_name = 'Alicia Keys'
            row.entity_name_id = '2'
        if row.entity_id == 'RL2013D04E149':
            row.entity_name = 'The Beatles'
            row.entity_name_id = '3'
        if row.entity_id == 'RL2013D04E151':
            row.entity_name = 'Led Zeppelin'
            row.entity_name_id = '4'
        if row.entity_id == 'RL2013D04E152':
            row.entity_name = 'Aerosmith'
            row.entity_name_id = '5'
        if row.entity_id == 'RL2013D04E153':
            row.entity_name = 'Bon Jovi'
            row.entity_name_id = '6'
        if row.entity_id == 'RL2013D04E155':
            row.entity_name = 'U2'
            row.entity_name_id = '7'
        if row.entity_id == 'RL2013D04E159':
            row.entity_name = 'AC/DC'
            row.entity_name_id = '8'
        if row.entity_id == 'RL2013D04E161':
            row.entity_name = 'The Wanted'
            row.entity_name_id = '9'
        if row.entity_id == 'RL2013D04E162':
            row.entity_name = 'Maroon 5'
            row.entity_name_id = '10'
        if row.entity_id == 'RL2013D04E164':
            row.entity_name = 'Coldplay'
            row.entity_name_id = '11'
        if row.entity_id == 'RL2013D04E166':
            row.entity_name = 'Lady Gaga'
            row.entity_name_id = '12'
        if row.entity_id == 'RL2013D04E167':
            row.entity_name = 'Madonna'
            row.entity_name_id = '13'
        if row.entity_id == 'RL2013D04E169':
            row.entity_name = 'Jennifer Lopez'
            row.entity_name_id = '14'
        if row.entity_id == 'RL2013D04E175':
            row.entity_name = 'Justin Bieber'
            row.entity_name_id = '15'
        if row.entity_id == 'RL2013D04E185':
            row.entity_name = 'Shakira'
            row.entity_name_id = '16'
        if row.entity_id == 'RL2013D04E194':
            row.entity_name = 'PSY'
            row.entity_name_id = '17'
        if row.entity_id == 'RL2013D04E198':
            row.entity_name = 'The Script'
            row.entity_name_id = '18'
        if row.entity_id == 'RL2013D04E206':
            row.entity_name = 'Whitney Houston'
            row.entity_name_id = '19'
        if row.entity_id == 'RL2013D04E207':
            row.entity_name = 'Britney Spears'
            row.entity_name_id = '20'
    tweet_data_frame.to_pickle('music.pkl')

    return tweet_data_frame


def read_music(music_file):
    music_data_frame = pd.DataFrame(pd.read_pickle(music_file), columns=['url', 'text',
                                                                         'id', 'author',
                                                                         'entity_id', 'tweet_url',
                                                                         'language', 'timestamp',
                                                                         'urls',
                                                                         'extended_urls',
                                                                         'md5_extended_urls',
                                                                         'is_near_duplicate_of',
                                                                         'set', 'tokenized', 'text_vec', 'set_vec',
                                                                         'wrd_set', 'entity_name', 'entity_name_id'])
    music_data_frame = music_data_frame.drop(
        columns=['url', 'tweet_url', 'timestamp', 'urls', 'extended_urls', 'md5_extended_urls',
                 'is_near_duplicate_of', 'tokenized'])

    return music_data_frame


# model = Word2Vec(tweets.set, min_count=1)
# TODO: watch k-means clustering
# TODO: check how to do similarity check with semantic similarity
# TODO: cluster word2vec data
# TODO: fetch labels to tweets and use those with MUSIC label :DONE
# TODO: label data from info files  :DONE
# TODO: Divide Music data from other data for base journal :DONE
tweets = read_music('music.pkl')
model = api.load("glove-twitter-25")


def text2vec(text):
    return np.mean([model[x] for x in text.split() if x in model.vocab], axis=0).reshape(1, -1)

tweets['vectors'] =''
tweets['vectors'] = text2vec(tweets.wrd_set)

# tweets = read_tweets('tweets-vector-2.pkl')

X = np.concatenate(tweets['set_vec'].values)
kmeans = KMeans(n_clusters=20)
kmeans.fit(X.reshape(-1, 1), y=tweets.entity_name)

# model = api.load("glove-twitter-25")  # load glove vectors
set_vec = tweets.set_vec.to_numpy
text_vec = tweets.text_vec.to_numpy
# print(model.most_similar("cat"))
