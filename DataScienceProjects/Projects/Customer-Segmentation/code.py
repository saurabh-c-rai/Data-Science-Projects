# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

offers = pd.read_excel(path, sheet_name=0)
transactions = pd.read_excel(path, sheet_name=1)
transactions['n'] = 1
df = offers.merge(transactions,how='outer')
df.head()

#transactions.columns,offers.columns


# --------------
matrix = df.pivot_table(index='Customer Last Name',columns='Offer #',values='n')
matrix.fillna(0, inplace=True)
matrix.reset_index(inplace=True)
matrix.head()


# --------------
# import packages
from sklearn.cluster import KMeans

km = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10 ,random_state=0)
matrix['cluster'] = km.fit_predict(matrix[matrix.columns[1:]])
matrix.head()


# --------------
# import packages
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0)
matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
clusters = pd.DataFrame(matrix.iloc[:,[0,33,34,35]])
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')


# --------------
data = clusters.merge(transactions,on='Customer Last Name')
data = offers.merge(data)
champagne = {}

for i in range(0,5):
    new_df = data[data['cluster']==i]
    counts = new_df['Varietal'].value_counts(ascending=False)
    if counts.index[0] == 'Champagne':
        champagne[i] = counts[0]
cluster_champagne = max(champagne, key=champagne.get)
print(cluster_champagne)


# --------------
discount = {}

for i in range(0,5):
    new_df = data[data['cluster']==i]
    counts = new_df['Discount (%)'].sum()/len(new_df)
    discount[i] = counts
cluster_discount = max(discount, key=discount.get)
print(cluster_discount)


