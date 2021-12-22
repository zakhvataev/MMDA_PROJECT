import pandas as pd
import numpy as np
import hdbscan
from sklearn.cluster import MiniBatchKMeans

import datetime as dt
import os

model_path = os.environ.get("INPUT_DATASET", "")
model_path = os.path.join(model_path, "data.csv")
df = pd.read_csv('data.csv', sep=';', dtype = {'party_rk': str})
upd_df = df[['party_rk', 'event_name']].drop_duplicates(subset = None)
upd_df['value'] = 1
upd_df1 = upd_df.pivot(index="party_rk", columns="event_name", values = 'value').fillna(0)

print("Clustering starting", datetime.datetime.now())
mbk = MiniBatchKMeans(init='k-means++', n_clusters=30, batch_size=200, n_init=30)
print("Clustering starting2", datetime.datetime.now())
clusters = mbk.fit_predict(df.tocsr())
print("Clustering done_", datetime.datetime.now())
from numpy import savetxt
savetxt('clusters1.csv', mbk.cluster_centers_, delimiter=';')



rangE = 0.15
clustering = hdbscan.HDBSCAN(alpha =rangE, min_cluster_size = 900, algorithm = 'boruvka_balltree', core_dist_n_jobs = -1, leaf_size = 14, min_samples = 1).fit(upd_df1)
clustirs = len(np.unique(clustering.labels_))
print(dt.datetime.now())
np.savetxt("clusters_hdbscan.csv", clustering.labels_, delimiter=",")
