#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import contractions
import matplotlib.pyplot as plt


# In[22]:


#reading above created file
df = pd.read_csv('data/reviews.csv')
print(f'Data has {df.shape[0]} rows and {df.shape[1]} columns')
df.head(5)


# In[23]:


#bad reviews - 0
df.loc[df['overall'].isin([1, 2]), 'overall'] = 0

#Good reviews - 1
df.loc[df['overall'].isin([4, 5]), 'overall'] = 1


# In[24]:


#good and bad reviews distribution
df['overall'].value_counts()


# In[25]:


#pie chart for review distribution
plt.pie(df['overall'].value_counts(), labels = ['good','bad'],autopct = '%1.2f%%')
plt.show()


# In[26]:


#dropping non ml attributes
non_ml_attr = ['reviewerID','asin','reviewerName', 'unixReviewTime', 'reviewTime', 'summary']
df.drop(non_ml_attr, axis=1, inplace=True)
df.head(5)


# In[27]:


import ast

#the helpful column is interpreted as string. converting it to array.
df['helpful'] = df['helpful'].apply(ast.literal_eval)

print(f'''Number of records where 2nd value of "helpful" is less than 1st value of "helpful" : {df[df['helpful'].str[0] > df['helpful'].str[1]].shape[0]}''')
print(f'''\nNumber of records where 2nd value of "helpful" is 0 but not 1st value of "helpful" : {df[(df['helpful'].str[1] == 0) & (df['helpful'].str[0] != 0)].shape[0]}''')


# In[28]:


#creating a helpful ratio : 1st value / 2nd value

df['helpful_ratio'] = df['helpful'].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0)
df.drop(['helpful'], axis = 1, inplace = True)
df.head(5)


# In[29]:


#dropping nulls
print('Before:')
print('- No. of rows:',df.shape[0])
print('- Missing value report:')
display(df.isna().sum())

df.dropna(inplace = True)

print('After:')
print('- No. of rows:',df.shape[0])
print('- Missing value report:')
display(df.isna().sum())


# In[30]:


# create review length column
df['review_length'] = df['reviewText'].apply(lambda x: len(x))
df.head()


# ### Cleaning and tranforming

# In[31]:


stop_words = set(stopwords.words('english'))

#creating a list of words that might actually help in sentiment analysis 
#and removing them from stopwords
x = ['few', 'once', 'same', 'below', 'above', 'during','over', 'after', 'most','before', 'just', 'against','very','no','which','where','what','nor','whom','why','when','down','but', 'not']
for i in x:
    stop_words.remove(i)
print(stop_words)


# In[32]:


def preprocess_text(text):
    '''
    preprocessing the required text column to convert case, remove number, remove contractions and stopwords
    '''
    # Convert to lower case
    text = text.lower()
    
    ## add space inbetween numbers and letters (e.g. 5mg to 5 mg, 17yo to 17 yo)
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
    
    #remove numbers
    text = re.sub(r'\d+', '', text)

    # Expand contractions (e.g., "can't" to "can not")
    text = contractions.fix(text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Apply stopwords list
    #stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    

    return text


# In[33]:


#preprocessing text
df['preprocessed_text'] = df['reviewText'].apply(lambda x: preprocess_text(x))
df['preprocessed_text'].head(3)


# In[34]:


#getting list of rare words - words that occur only once.
#This might help in eliminating typos
text = ' '.join(df['preprocessed_text'])
text = text.split()
freq_comm = pd.Series(text).value_counts()
rare = freq_comm[freq_comm.values == 1]
rare.index.tolist()[:10]


# In[35]:


def remove_rare(text):
    '''
    function to remove rare words
    '''
    text = ' '.join([word for word in text.split() if word not in rare])
    return text


# In[36]:


#removing the above rare words:
df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: remove_rare(x))
df['preprocessed_text'].head(3)


# In[37]:


#creating list of words.
#each element is a list of words in a sentence.
words = []
for i in df['preprocessed_text'].values:
    words.append(i.split())
print(words[:3])


# In[38]:


#creating review preprocessed text column - as there are a few texts of length 0
df['processed_len'] = df['preprocessed_text'].apply(lambda x: len(x))

#printing rows where length of preprocessed text is 0
df[df['processed_len'] == 0]


# In[39]:


#removing those
print('Before:', df.shape[0])
df = df[df['processed_len'] != 0]
print('After:', df.shape[0])


# In[40]:


print('Saving the preprocessed data...')
df.to_csv('data/preprocessed.csv', index=False)

