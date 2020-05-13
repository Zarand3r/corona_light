import numpy as np
from sklearn.datasets import fetch_mldata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from sklearn.manifold import TSNE
# import tensorflow.examples.tutorials.mnist.input_data as input_data




datafile = '' #get this either from command line argument or encapslate into function that can be imported in other files
data = np.loadtxt(open(datafile, 'rb'), delimiter=",", skiprows=1)
X = data[:, 1:-1]
y = data[:, -1]


print (X.shape, y.shape)


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

print ('Size of the dataframe: {}'.format(df.shape))

rndperm = np.random.permutation(df.shape[0])


n_sne = 7000

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

# Create the figure
fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1, 1, 1, title='TSNE' )
# Create the scatter
ax.scatter(
    x=df_tsne['x-tsne'], 
    y=df_tsne['y-tsne'], 
    c=df_tsne['label'], 
    cmap=plt.cm.get_cmap('Paired'), 
    alpha=0.15)
plt.show()

# pca_50 = PCA(n_components=50)
# pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
# print ('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))