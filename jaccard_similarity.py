# Reading from tweets adn calculating jaccard similarity for each row

# imports

import time
from numpy import save
import pandas as pd

'''
for val in list(combinations(range(daata(df)), 2)):
    firstlist = data.iloc[val[0], 1]
    secondlist = data.iloc[val[1], 1]

    value = round(lexical_overlap(firstlist, secondlist), 2)

    print(f"{data.iloc[val[0], 0]} and {data.iloc[val[1], 0]}'s value is: {value}")



'''
NUM_PROC = 4

data = pd.read_excel('tweet_words.xlsx', 'Sheet1')
data = data.drop(columns=['cleaned', 'test'])
data['words'] = data['words'].str.split()
simil_mat = pd.DataFrame()

x = dict(zip(data.tweet_id, data.words))

i = 1
for k, v in x.items():
    s1 = set(str(v).lower().split())
    t1 = time.time()
    print(i)
    i += 1
    for k2, v2 in x.items():
        s2 = set(str(v2).lower().split())
        try:
            simil_mat.loc[k, k2] = float(len(s1.intersection(s2)) / len(s1.union(s2)))
        except:
            simil_mat.loc[k, k2] = 0

    if i % 100 == 0:
        simil_mat.to_pickle('jaccard_similarity.pkl')
        n = simil_mat.to_numpy()
        save('jaccard_similarity.npy', n)

    t2 = time.time()
    print('time:', t2 - t1)
simil_mat.to_pickle('jaccard_similarity.pkl')
n = simil_mat.to_numpy()
save('jaccard_similarity.npy', n)
