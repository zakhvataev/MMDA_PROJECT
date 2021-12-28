import pandas as pd
import numpy as np
df = pd.read_csv('data.csv', sep=';', dtype = {'party_rk': str})
upd_df = df[['party_rk', 'event_name']].drop_duplicates(subset = None)
categories = pd.read_csv('category_catalog.csv', sep=';', )
categories = categories.set_index('new_original_name', drop=True)
categories = categories[['new_group_det1']].to_dict()
upd_df = upd_df.replace({"event_name": categories['new_group_det1']})
