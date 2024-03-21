#!/usr/bin/env python
# coding: utf-8

# In[7]:


import gensim
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# In[8]:


df = pd.read_csv('data/preprocessed.csv')
df


# In[9]:


def create_w2v(embedding_dim, words, window, epochs, sg):
    '''
    creating word2vec or skipgram.
    '''
    #Creating Word2Vec
    w2v_model = gensim.models.Word2Vec(words , vector_size = embedding_dim , window = window , min_count = 1, epochs = epochs, sg = sg)
    print(f"- {'Word2Vec' if(sg==0) else 'Skipgram'} Created")
    print(f'- Vocabulary count: {len(w2v_model.wv)}')
    print(f'''- Similar words for word "great:\n"{w2v_model.wv.most_similar('great')}''')
    
    return w2v_model
    

def get_sentence_embedding(sent, model, flag):
    '''
    create embeddings by calculating mean of vectors of words in each review (preprocessed_text)
    '''
    list_vectors = []
    for word in sent:
        if(flag):
            #indicates fasttext
            vector = model.get_word_vector(word)
        else:
            vector = model.wv[word]
        list_vectors.append(vector)
    mean_vector = np.array(list_vectors).mean(axis=0)
    return mean_vector


# Creating "embeddings" column
def get_embedding_cols(df, embedding_dim, model, flag):
    '''
    returns df with embedded columns. flag indicates if its fasttext
    '''
    #df['embeddings'] = df['preprocessed_text'].apply(lambda x: get_sentence_embedding(x.split(), model))
    df['embeddings'] = df['preprocessed_text'].apply(lambda x: get_sentence_embedding(x.split(), model, flag))
    

    #creating a column for each vector in embedding - 100 columns
    cols = [f'e_{i}' for i in range(1, embedding_dim + 1)]
    df[cols] = pd.DataFrame(df['embeddings'].tolist(), index= df.index)
    print('- Embeddings are created.')
    return df


# ### Fasttext

# In[10]:


df.to_csv('data/text_label.txt', columns = ['preprocessed_text'], header = None, index = False)


# In[11]:


import fasttext

model = fasttext.train_unsupervised('data/text_label.txt', dim = 100)

print(f'FASTTEXT {model.dim} VECTOR EMBEDDING DIMENSIONS:')
print(f'=========================================')

ft_df_100 = get_embedding_cols(df, 100, model, True)

#writing to a new file
ft_df_100.to_csv('data/fasttext_100.csv', index = False)

print('- Fasttext embeddings Created')
print(f'- Vocabulary count: {len(model.words)}')
print(f'''- Similar words for word "great:\n"{model.get_nearest_neighbors('great', k=10)}''')

model.save_model('models/fasttext_model.bin')


# ### Word2Vec - CBOW

# In[14]:


words = []
for i in df['preprocessed_text'].values:
    words.append(i.split())
words[:3]


# In[15]:


print(f'\nWORD2VEC 100 VECTOR EMBEDDING DIMENSIONS:')
print(f'=========================================')

#word2vec
cbow_model = create_w2v(100, words, 7, 50, sg = 0)

df1 = df.copy()

#creating embedding columns
df1 = get_embedding_cols(df1, 100, cbow_model, False)
df1.to_csv(f'word2vec_100.csv', index = False)

cbow_model.save('models/cbow.model')


# ### Word2Vec - Skipgram

# In[16]:


print(f'\nSKIPGRAM 100 VECTOR EMBEDDING DIMENSIONS:')
print(f'=========================================')

#word2vec
sg_model = create_w2v(100, words, 7, 50, sg = 1)

df1 = df.copy()

#creating embedding columns
df1 = get_embedding_cols(df1, 100, sg_model, False)
df1.to_csv(f'skipgram_100.csv', index = False)

sg_model.save('models/skipgram.model')


# In[17]:


print('Embeddings are created.')


# In[ ]:




