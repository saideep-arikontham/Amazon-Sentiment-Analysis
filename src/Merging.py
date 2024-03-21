#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

df1 = pd.read_json('data/reviews_Video_Games_5.json', lines = True)
df2 = pd.read_json('data/reviews_Patio_Lawn_and_Garden_5.json', lines = True)
df3 = pd.read_json('data/reviews_Automotive_5.json', lines = True)

#merging dataframes
final_df = pd.concat([df1, df2, df3])


#Writing to a new file removing overall rating=3 records to later group (1, 2) as bad and (4, 5) as good
final_df = final_df[final_df['overall'].isin([1,2,4,5])]
print(f'We have a total of {final_df.shape[0]} reviews')
print('Saving the combined reviews to new file...')
final_df.to_csv('data/reviews.csv', index=False)
print('Saved as "data/reviews.csv"')

