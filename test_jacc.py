import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

df = pd.read_excel('tweets_words_en.xlsx', 'Sheet1')
df = df.dropna()
n = df.to_numpy()
x = 0
f = np.zeros(shape=(25540, 25540))
t0 = time()
list_columns = []
for i in tqdm(n[:, 8]):
    list_rows = []
    y = 0
    for j in n[:, 8]:
        # l = []
        if f[x, y] == 0:
            if i == j:
                # l.append(float(len(set(i).intersection(set(j))) / len(set(i).union(set(j)))))
                # f[x][y] = float(len(set(i).intersection(set(j))) / len(set(i).union(set(j))))
                f[x][y] = 1
                f[y][x] = 1
            elif i != j:
                # l.append(float(len(set(i).intersection(set(j))) / len(set(i).union(set(j)))))
                f[x][y] = float(len(set(i).intersection(set(j))) / len(set(i).union(set(j))))
                f[y][x] = f[x][y]
                if f[x][y] < 0.8:
                    f[x][y] = 10
                    f[y][x] = f[x][y]
        else:
            # f[x,y]=f[x,y]
            pass
        y += 1
        # list_rows.append(l)
    # list_columns.append(list_rows)
    # list_columns[x][:] = np.transpose(list_columns)
    # list_columns[x] = np.transpose(list_rows)
    # f[:,x] = np.transpose(f[x,:])
    x += 1
    # if x > 255:
    #   break
    # simil_mat = pd.DataFrame(list_columns)
    if f[25539, 25539] != 0:
        break
f[f == 10] = 0
t1 = time()
print("255 go in ", t1 - t0)
s = pd.DataFrame(f)
s.to_pickle('jaccard_similarity.pkl')
print("salavat")
np.save('jaccard_similarity.npy', f)
