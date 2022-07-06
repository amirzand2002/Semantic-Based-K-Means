# clean tweet and get all word cluster for every single word in tweets text
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

stopword_en = nltk.corpus.stopwords.words('english')
stopword_es = nltk.corpus.stopwords.words('spanish')
stopword = stopword_en + stopword_es

tqdm.pandas(desc="progress-bar")
t0 = time.time()
# ---------------------read from tweet from all_replab.xlsx--------------------------------#
data_test = pd.read_excel('all_replab.xlsx', 'Sheet1')
tweets = pd.DataFrame(data_test, columns=['tweet_url', 'tweet_text',
                                          'tweet_id', 'author',
                                          'entity_id', 'tweet_url',
                                          'language', 'timestamp',
                                          'urls',
                                          'extended_urls',
                                          'md5_extended_urls',
                                          'is_near_duplicate_of'])
tweets['words'] = ''
del data_test
tweets = tweets[tweets['tweet_text'] != 'none']
tweets = tweets[tweets['tweet_text'] != 'None']
tweets = tweets[tweets['tweet_text'] != 'Suspended']
tweets = tweets[tweets['tweet_text'] != 'Noneccount Suspended']
# tweets_en = tweets[tweets['language'] == 'EN']
# tweets_es = tweets[tweets['language'] == 'ES']
tweets = tweets.drop(columns=['tweet_url', 'timestamp', 'urls', 'extended_urls', 'md5_extended_urls',
                              'is_near_duplicate_of'])
tweets = tweets[tweets.entity_id.isin(['RL2013D04E145', 'RL2013D04E146', 'RL2013D04E149', 'RL2013D04E151',
                                       'RL2013D04E152', 'RL2013D04E153',
                                       'RL2013D04E155', 'RL2013D04E159', 'RL2013D04E161', 'RL2013D04E162',
                                       'RL2013D04E164',
                                       'RL2013D04E166', 'RL2013D04E167', 'RL2013D04E169', 'RL2013D04E175',
                                       'RL2013D04E185',
                                       'RL2013D04E194', 'RL2013D04E198', 'RL2013D04E206', 'RL2013D04E207'])]
tweets['entity_name'] = ''
tweets['entity_name_id'] = ''
# read entity_id and assign entity name and name id (label)  to it
tweets.loc[tweets['entity_id'] == 'RL2013D04E145', 'entity_name'] = 'Adele'
tweets.loc[tweets['entity_id'] == 'RL2013D04E145', 'entity_name_id'] = '1'
tweets.loc[tweets['entity_id'] == 'RL2013D04E146', 'entity_name'] = 'Alicia Keys'
tweets.loc[tweets['entity_id'] == 'RL2013D04E146', 'entity_name_id'] = '2'
tweets.loc[tweets['entity_id'] == 'RL2013D04E146', 'entity_name'] = 'Alicia Keys'
tweets.loc[tweets['entity_id'] == 'RL2013D04E146', 'entity_name_id'] = '2'

tweets.loc[tweets['entity_id'] == 'RL2013D04E149', 'entity_name'] = 'The Beatles'
tweets.loc[tweets['entity_id'] == 'RL2013D04E149', 'entity_name_id'] = '3'

tweets.loc[tweets['entity_id'] == 'RL2013D04E151', 'entity_name'] = 'Led Zeppelin'
tweets.loc[tweets['entity_id'] == 'RL2013D04E151', 'entity_name_id'] = '4'

tweets.loc[tweets['entity_id'] == 'RL2013D04E152', 'entity_name'] = 'Aerosmith'
tweets.loc[tweets['entity_id'] == 'RL2013D04E152', 'entity_name_id'] = '5'

tweets.loc[tweets['entity_id'] == 'RL2013D04E153', 'entity_name'] = 'Bon Jovi'
tweets.loc[tweets['entity_id'] == 'RL2013D04E153', 'entity_name_id'] = '6'

tweets.loc[tweets['entity_id'] == 'RL2013D04E155', 'entity_name'] = 'U2'
tweets.loc[tweets['entity_id'] == 'RL2013D04E155', 'entity_name_id'] = '7'

tweets.loc[tweets['entity_id'] == 'RL2013D04E159', 'entity_name'] = 'AC/DC'
tweets.loc[tweets['entity_id'] == 'RL2013D04E159', 'entity_name_id'] = '8'

tweets.loc[tweets['entity_id'] == 'RL2013D04E161', 'entity_name'] = 'The Wanted'
tweets.loc[tweets['entity_id'] == 'RL2013D04E161', 'entity_name_id'] = '9'

tweets.loc[tweets['entity_id'] == 'RL2013D04E162', 'entity_name'] = 'Maroon 5'
tweets.loc[tweets['entity_id'] == 'RL2013D04E162', 'entity_name_id'] = '10'

tweets.loc[tweets['entity_id'] == 'RL2013D04E164', 'entity_name'] = 'Coldplay'
tweets.loc[tweets['entity_id'] == 'RL2013D04E164', 'entity_name_id'] = '11'

tweets.loc[tweets['entity_id'] == 'RL2013D04E166', 'entity_name'] = 'Lady Gaga'
tweets.loc[tweets['entity_id'] == 'RL2013D04E166', 'entity_name_id'] = '12'

tweets.loc[tweets['entity_id'] == 'RL2013D04E167', 'entity_name'] = 'Madonna'
tweets.loc[tweets['entity_id'] == 'RL2013D04E167', 'entity_name_id'] = '13'

tweets.loc[tweets['entity_id'] == 'RL2013D04E169', 'entity_name'] = 'Jennifer Lopez'
tweets.loc[tweets['entity_id'] == 'RL2013D04E169', 'entity_name_id'] = '14'

tweets.loc[tweets['entity_id'] == 'RL2013D04E175', 'entity_name'] = 'Justin Bieber'
tweets.loc[tweets['entity_id'] == 'RL2013D04E175', 'entity_name_id'] = '15'

tweets.loc[tweets['entity_id'] == 'RL2013D04E185', 'entity_name'] = 'Shakira'
tweets.loc[tweets['entity_id'] == 'RL2013D04E185', 'entity_name_id'] = '16'

tweets.loc[tweets['entity_id'] == 'RL2013D04E194', 'entity_name'] = 'PSY'
tweets.loc[tweets['entity_id'] == 'RL2013D04E194', 'entity_name_id'] = '17'

tweets.loc[tweets['entity_id'] == 'RL2013D04E198', 'entity_name'] = 'The Script'
tweets.loc[tweets['entity_id'] == 'RL2013D04E198', 'entity_name_id'] = '18'

tweets.loc[tweets['entity_id'] == 'RL2013D04E206', 'entity_name'] = 'Whitney Houston'
tweets.loc[tweets['entity_id'] == 'RL2013D04E206', 'entity_name_id'] = '19'

tweets.loc[tweets['entity_id'] == 'RL2013D04E207', 'entity_name'] = 'Britney Spears'
tweets.loc[tweets['entity_id'] == 'RL2013D04E207', 'entity_name_id'] = '20'

t1 = time.time()
print("time of reading from xlsx file ", t1 - t0)

# -----------------preprocess text--------------------- #
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(text):
    return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize(text)]


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


tweets['cleaned'] = ''
w_tokenizer = TweetTokenizer()
tweets['tweet_text'] = tweets['tweet_text'].apply(str)
tweets['cleaned'] = tweets['tweet_text'].str.lower().replace('[^\w\s]', ' ').replace('\s\s+', ' ')
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: re.sub(r'\s\s+', '', x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: p.clean(x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: lemmatize_text(x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: remove_stopwords(x))

word_set = pd.read_excel('word_set .xlsx', index_col=[0])
word_set = word_set[word_set.set != '[]']
tweets['test'] = tweets['cleaned'].to_list()

t2 = time.time()
print("time of preprocessing ", t2 - t1)
# finding words from word set of flicker and strong them in tweets.words column
for index, row in tweets.iterrows():
    for item in tweets.test[index]:
        tweets.words[index] += str(word_set[word_set['word'] == item]['set'].values) + ''
tweets.to_excel("tweet_words.xlsx")
t3 = time.time()
print("time of finding word group ", t3 - t2)
