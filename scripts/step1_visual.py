import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
model_path = os.environ.get("INPUT_DATASET", "")
model_path = os.path.join(model_path, "data.csv")
df = pd.read_csv('data.csv', sep=';', dtype = {'party_rk': str})
data = df[['party_rk', 'event_name']].drop_duplicates(subset = None)
data['value'] = 1
data = upd_df.pivot(index="party_rk", columns="event_name", values = 'value').fillna(0)

def try_different_clusters(K, data):
       
    cluster_values = list(range(1, K+1))
    inertias=[]
    
    for c in cluster_values:
        model = MiniBatchKMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)
        print(c)
    
    return inertias
outputs = try_different_clusters(40, data)
distances = pd.DataFrame({"clusters": list(range(1, 41)),"sum of squared distances": outputs})

figure = go.Figure() #number of clusters
figure.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

figure.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),                  
                  xaxis_title="Number of clusters",
                  yaxis_title="Sum of squared distances",
                  title_text="Finding optimal number of clusters using elbow method")
figure.show()

mbk = MiniBatchKMeans(init='k-means++', n_clusters=20, batch_size=200, n_init=20)
clusters = mbk.fit_predict(data)
data['cluster'] = clusters
random_elements = features_copy.groupby('cluster').sample(frac=.3)
X_embedded = TSNE(random_state=0, perplexity=50, n_iter=1000).fit_transform(random_elements.loc[:, random_elements.columns != 'cluster'])
X_embedded.shape
clusters_vectors = pd.DataFrame({'x':X_embedded[:,0], 'y':X_embedded[:,1], 'z':clusters})
sns.set(rc={'figure.figsize':(14, 16)})
scatter = sns.scatterplot(x=X_embedded[:,0],
                          y=X_embedded[:,1],
                          hue=random_elements.cluster,
                          legend='full',
                          palette=palette) #plane projection
