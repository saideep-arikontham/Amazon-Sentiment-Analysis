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


# # Using CNN

# ### CBOW

# In[3]:


from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import set_random_seed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

set_random_seed(80)

X = wv.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = wv['overall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

cw = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
class_weights = {np.unique(y_train)[0]: cw[0], np.unique(y_train)[1]: cw[1]} 

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # Adjust layer size based on your needs
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate = 0.3), metrics=['accuracy'] )


X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)


model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test), class_weight=class_weights)


yhat_probs = model.predict(X_test, verbose=0)
yhat_probs = yhat_probs[:, 0]
y_pred = pd.Series(yhat_probs).apply(lambda x: 1 if x > 0.5 else 0)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('figs/cnn_cbow_cm.png')
plt.show()

accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred)

roc_auc = get_roc_auc(y_test, yhat_probs, 'figs/cnn_cbow_roc.png')

scores_data.append(['CNN', 'CBOW']+[accuracy, precision, recall, f1, roc_auc])


# ### Skipgram

# In[4]:


set_random_seed(24)

X = sg.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = sg['overall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # Adjust layer size based on your needs
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate = 0.1), metrics=['accuracy'] )


X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)


model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), class_weight=class_weights)


yhat_probs = model.predict(X_test, verbose=0)
yhat_probs = yhat_probs[:, 0]
y_pred = pd.Series(yhat_probs).apply(lambda x: 1 if x > 0.5 else 0)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('figs/cnn_skipgram_cm.png')
plt.show()

accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred)

roc_auc = get_roc_auc(y_test, yhat_probs, 'figs/cnn_skipgram_roc.png')

scores_data.append(['CNN', 'Skipgram']+[accuracy, precision, recall, f1, roc_auc])


# ### Fasttext

# In[5]:


set_random_seed(22)

X = ft.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = ft['overall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # Adjust layer size based on your needs
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer= Adam(learning_rate = 0.01), metrics=['accuracy'] )


X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)


model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), class_weight=class_weights)


yhat_probs = model.predict(X_test, verbose=0)
yhat_probs = yhat_probs[:, 0]
y_pred = pd.Series(yhat_probs).apply(lambda x: 1 if x > 0.5 else 0)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('figs/cnn_fasttext_cm.png')
plt.show()

accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred)

roc_auc = get_roc_auc(y_test, yhat_probs, 'figs/cnn_fasttext_roc.png')

scores_data.append(['CNN', 'Fasttext']+[accuracy, precision, recall, f1, roc_auc])


# In[6]:


scores = pd.DataFrame(data = scores_data, columns = ['model', 'embedding','accuracy','precision','recall','f1 score','roc auc'])
scores


# In[7]:


scores.to_csv('data/CNN_performance.csv', index = False)


# In[ ]:




