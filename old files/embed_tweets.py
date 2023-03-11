# calculate vector for both tweet text and flickr word set of tweets word
# imports
import numpy as np
import gensim.downloader as api
import pandas as pd
from tqdm import tqdm
import preprocessor as p
from nltk.tokenize import TweetTokenizer
import nltk
import re
from gensim.parsing.preprocessing import remove_stopwords

lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()


def lemmatize_text(text):
    return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize(text)]


# read tweets-set.pkl
tweets = pd.DataFrame(pd.read_pickle('tweets_set.pkl'), columns=['url', 'text',
                                                                 'id', 'author',
                                                                 'entity_id', 'tweet_url',
                                                                 'language', 'timestamp',
                                                                 'urls',
                                                                 'extended_urls',
                                                                 'md5_extended_urls',
                                                                 'is_near_duplicate_of',
                                                                 'set', 'tokenized'])
# drop unnecessary column
tweets = tweets.drop(columns=['tweet_url', 'timestamp', 'urls', 'extended_urls', 'md5_extended_urls',
                              'is_near_duplicate_of', 'tokenized'])
# load pre-train glove vectors for twitter
model = api.load("glove-twitter-25")

tweets['text_vec'] = ''
tweets['set_vec'] = ''
# make a column for vectors
# calculate vector for both tweet text and flickr word set of tweets word
for index, row in tweets.iterrows():
    if index % 1000 == 1:
        print(index)
    tweets.text_vec[index] = np.zeros(25)
    tweets.set_vec[index] = np.zeros(25)
    a = row.text.split()
    b = row.set.split()
    for item in a:
        try:

            tweets.text_vec[index] += model.get_vector(item)
        except:
            continue
    for item in b:
        try:

            tweets.set_vec[index] += model.get_vector(item)
        except:
            continue

# TODO:fasttext-wiki-news-subwords-300 , glove twiiter 200 take a lot of RAM
# TODO: JUST TRY TO DEAL WITH MUSIC ENTITY FOR TIME
tweets.to_pickle('tweets-vector-2.pkl')
print(model.most_similar("cat"))
