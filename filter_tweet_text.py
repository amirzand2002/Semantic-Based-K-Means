# clean tweet and get all word cluster for every single word in tweets text
#imports
import pandas as pd
import nltk
from nltk.corpus import stopwords
import preprocessor as p
import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import TweetTokenizer
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from itertools import chain

tqdm.pandas(desc="progress-bar")
t0 = time.time()
# read from all_replab.xlsx
data_test = pd.read_excel('all_replab.xlsx', 'Sheet1', ).values
tweets = pd.DataFrame(data_test, columns=['url', 'text',
                                          'id', 'author',
                                          'entity_id', 'tweet_url',
                                          'language', 'timestamp',
                                          'urls',
                                          'extended_urls',
                                          'md5_extended_urls',
                                          'is_near_duplicate_of'])
tweets['words'] = ''
tweets = tweets[tweets['text'] != 'none']
tweets = tweets[tweets['text'] != 'None']
tweets = tweets[tweets['text'] != 'Suspended']
tweets = tweets[tweets['text'] != 'Noneccount Suspended']
tweets_en = tweets[tweets['language'] == 'EN']
tweets_es = tweets[tweets['language'] == 'ES']
t1 = time.time()
print("time of reading from xlsx file ", t1 - t0)

# preprocess text
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = TweetTokenizer()
def lemmatize_text(text):
    return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize(text)]
stop_words = set(stopwords.words('english'))
tweets['text'] = tweets['text'].apply(str)
for index, row in tqdm(tweets.iterrows()):
    row.text = p.clean(row.text)
    row.text = remove_stopwords(row.text)
    row.text = row.text.lower().replace('[^\w\s]', ' ').replace('\s\s+', ' ')
    row.text = re.sub(r'[^\w\s]', ' ', row.text)
    row.text = lemmatize_text(row.text)

tweets.to_pickle('tweets.pkl')
tweets.to_excel('tweets.xlsx')
# save clean tweet in .pkl file
uni_set = set(chain(*tweets.text.to_list()))
t2 = time.time()
print("preprocessing", t2 - t1)


set_df = pd.DataFrame(uni_set, columns=['word'])
set_df['set'] = ''
payload = {}
headers = {}
these_regex = "<tag.*?>(.+?)</tag>"
pattern = re.compile(these_regex)

# get word cluster from flickr site
try:
    for i in tqdm(set_df.index):
        url = "https://www.flickr.com/services/rest/?method=flickr.tags.getRelated&api_key=9e9a14ab054552c65a4ea5e7033e7fc6&tag=&format=rest"
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
set_df.to_pickle('word_set.pkl')
set_df.to_excel('word_set.xlsx')
t4 = time.time()
print("saving", t4 - t3)
t5 = time.time()
print("total", t5 - t0)