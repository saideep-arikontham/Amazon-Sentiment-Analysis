#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
from performance_utils import calc_metrics, get_roc_auc
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

wv = pd.read_csv('data/word2vec_100.csv')
sg = pd.read_csv('data/skipgram_100.csv')
ft = pd.read_csv('data/fasttext_100.csv')


# In[2]:


scores_data = []


# # Using RNN

# ### CBOW

# In[3]:


from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from keras.optimizers import Adam
from keras.utils import set_random_seed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

set_random_seed(76)

# Load your data
X = wv.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = wv['overall']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

cw = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
class_weights = {np.unique(y_train)[0]: cw[0], np.unique(y_train)[1]: cw[1]} 

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))  # Adjust input_shape based on your data
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification

# Compile the model
model.compile(optimizer=Adam(learning_rate = 0.009), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#9
model.fit(X_train, y_train, epochs=9, batch_size=128, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('figs/rnn_cbow_cm.png')
plt.show()

y_pred_proba = model.predict(X_test)

accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred)

roc_auc = get_roc_auc(y_test, y_pred_proba, 'figs/rnn_cbow_roc.png')

scores_data.append(['RNN', 'CBOW']+[accuracy, precision, recall, f1, roc_auc])


# ### Skipgram

# In[4]:


set_random_seed(42)

# Load your data
X = sg.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = sg['overall']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)


model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))  # Adjust input_shape based on your data
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification

# Compile the model
model.compile(optimizer=Adam(lr = 0.6), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#9
model.fit(X_train, y_train, epochs=9, batch_size=128, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('figs/rnn_skipgram_cm.png')
plt.show()

y_pred_proba = model.predict(X_test)

accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred)
roc_auc = get_roc_auc(y_test, y_pred_proba, 'figs/rnn_skipgram_roc.png')

scores_data.append(['RNN', 'Skipgram']+[accuracy, precision, recall, f1, roc_auc])



# ### Fasttext

# In[5]:


set_random_seed(42)

# Load your data
X = ft.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = ft['overall']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)



model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1)))  # Adjust input_shape based on your data
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron for binary classification

# Compile the model
model.compile(optimizer=Adam(lr = 0.7), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
#13
model.fit(X_train, y_train, epochs=45, batch_size=128, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('figs/rnn_fasttext_cm.png')
plt.show()

y_pred_proba = model.predict(X_test)

accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred)
roc_auc = get_roc_auc(y_test, y_pred_proba, 'figs/rnn_fasttext_roc.png')


scores_data.append(['RNN', 'Fasttext']+[accuracy, precision, recall, f1, roc_auc])


# In[6]:


scores = pd.DataFrame(data = scores_data, columns = ['model', 'embedding','accuracy','precision','recall','f1 score','roc auc'])
scores


# In[7]:


scores.to_csv('data/RNN_performance.csv', index = False)


# In[ ]:




