import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

'''plot data so that we can decide what to do with them'''

'''First plot with given label '''
# Read the embeddings from the Excel file
data = pd.read_excel("tweets_with_emb.xlsx")
embeddings = np.array(data['embedding'].tolist())

# convert embeddings string to a numpy array
embeddings_array = np.array([list(map(float, emb.strip('[]').split())) for emb in embeddings], dtype=np.float32)

# perform TSNE on the numerical array
tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings_array)
entity_name_ids = np.array(data['entity_name_id'])

# Create a dictionary to map entity name ids to colors
id_color_dict = {id: np.random.rand(3, ) for id in entity_name_ids}

# Create a list of colors for each point based on its entity name id
point_colors = [id_color_dict[id] for id in entity_name_ids]

# Plot the scatter plot with colors
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=point_colors)
plt.title("Embeddings in 2D space")
plt.show()
######
'''plot second one with predicted label'''

entity_name_ids = np.array(data['label'])

# Create a dictionary to map entity name ids to colors
id_color_dict_2 = {id: np.random.rand(3, ) for id in entity_name_ids}

# Create a list of colors for each point based on its entity name id
point_colors = [id_color_dict_2[id] for id in entity_name_ids]

# Plot the scatter plot with colors
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=point_colors)
plt.title("Embeddings in 2D space")
plt.show()
