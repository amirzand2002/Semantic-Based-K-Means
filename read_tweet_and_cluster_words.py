# clean tweet and get all word cluster for every single word in tweets text
# imports
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
tweets_en = tweets[tweets['language'] == 'EN']
tweets_es = tweets[tweets['language'] == 'ES']
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


'''for index, row in tqdm(tweets.iterrows()):
    # row.text = p.clean(row.text)
    row.tweet_text = row.tweet_text.lower().replace('[^\w\s]', ' ').replace('\s\s+', ' ')
    row.tweet_text = p.clean(row.tweet_text)
    row.cleaned =[remove_stopwords(row.tweet_text)]
    # row.tweet_text = lemmatize_text(row.tweet_text)'''
tweets['cleaned'] = ''
w_tokenizer = TweetTokenizer()
tweets['tweet_text'] = tweets['tweet_text'].apply(str)
tweets['cleaned'] = tweets['tweet_text'].str.lower().replace('[^\w\s]', ' ').replace('\s\s+', ' ')
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: re.sub(r'\s\s+', '', x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: p.clean(x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: lemmatize_text(x))
tweets['cleaned'] = tweets['cleaned'].apply(lambda x: remove_stopwords(x))

# tweets.to_pickle('tweets.pkl')
tweets.to_excel('tweets.xlsx')
# save clean tweet in .pkl file
uni_set = set(chain(*tweets.cleaned.to_list()))
t2 = time.time()
print("preprocessing", t2 - t1)
# -------------------- end of pre processing tweets -----------------------------------
set_df = pd.DataFrame(uni_set, columns=['word'])
set_df['set'] = ''
payload = {}
headers = {}
these_regex = "<tag.*?>(.+?)</tag>"
pattern = re.compile(these_regex)

# ------------------------ get word cluster from flickr site ---------------------

for i in tqdm(set_df.index):
    try:
        url = "https://www.flickr.com/services/rest/?method=flickr.tags.getRelated&api_key=44f79721649ff0cba6022e25a3d78076&tag=&format=rest"
        url = url[:113] + set_df.word[i] + url[113:]
        # print(url)
        response = requests.request("GET", url)
        titles = re.findall(pattern, response.text)
        # print(titles)
        set_df.set[i] = titles
    except:
        print("An exception occurred")
t3 = time.time()
print("flickr time", t3 - t2)

# save word cluster in word_set.pkl
# set_df.to_pickle('word_set.pkl')
set_df.to_excel('word_set.xlsx')
t4 = time.time()
print("saving", t4 - t3)
t5 = time.time()
print("total", t5 - t0)
