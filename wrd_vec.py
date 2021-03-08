import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from tqdm import tqdm_notebook

tqdm_notebook().pandas()
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import seaborn as sb

sb.set(style='white')
pd.options.mode.chained_assignment = None
from tqdm import tqdm

tqdm.pandas()


# The function "text_to_wordlist" is from

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r'[^A-Za-z0-9^,!./\'+-=]', " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


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
    tweet_data_frame['text_set'] = tweet_data_frame.text + ' ' + tweet_data_frame.set
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


tweets = read_tweets('music.pkl')
tweets = tweets.drop(columns=['url', 'text', 'set', 'text_vec',
                              'wrd_set'])

words = tweets.text_set.progress_apply(text_to_wordlist)

import gensim.downloader as api

model = api.load('glove-twitter-25')


def text2vec(text):
    return np.mean([model[x] for x in text.split() if x in model.vocab], axis=0)


# vector_list = tweets.text_set.progress_apply(text2vec)
vector_list = words.progress_apply(text2vec)

words_filtered = [word for word in words if word in model.vocab]
word_vec_zip = zip(words_filtered, vector_list)
word_vec_dict = dict(word_vec_zip)
