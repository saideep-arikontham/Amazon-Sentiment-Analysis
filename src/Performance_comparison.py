#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

xgb = pd.read_csv('data/XGBoost_performance.csv')
cnn = pd.read_csv('data/CNN_performance.csv')
rnn = pd.read_csv('data/RNN_performance.csv')


# In[2]:


scores = pd.concat([xgb, cnn, rnn])
scores


# In[3]:


scores['method'] = scores['model'] + '-' + scores['embedding']
scores.drop(['model', 'embedding'], inplace = True, axis = 1)
scores


# In[4]:


metrics = ['accuracy', 'precision', 'recall', 'f1 score', 'roc auc']


# In[5]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 15))

for i in range(len(metrics)):
    sns.set_theme(style = 'darkgrid')
    best_acc = scores[['method', metrics[i]]].sort_values([metrics[i]])

    axes[i//3][i%3].bar(best_acc['method'], best_acc[metrics[i]], width = 0.4)
    axes[i//3][i%3].axhline(y = best_acc[metrics[i]].max(), color='r', linestyle='-', label = f'Best Score = {"{:.2f}".format(best_acc[metrics[i]].max())}')
    axes[i//3][i%3].axhline(y = best_acc[metrics[i]].min(), color='green', linestyle='--', label = f'Least Score = {"{:.2f}".format(best_acc[metrics[i]].min())}')

    axes[i//3][i%3].tick_params(labelrotation=45)
    axes[i//3][i%3].set_ylim(0, 1.2)
    axes[i//3][i%3].set_ylabel('Score')
    axes[i//3][i%3].set_title(f'{metrics[i]} score of different methods')
    axes[i//3][i%3].legend()

x = scores[scores['method'] == 'RNN-Fasttext'].to_numpy().tolist()[0][:5]
axes[1][2].bar(metrics, x, width = 0.4)
axes[1][2].set_xlabel('Metric')
axes[1][2].set_ylabel('Score')
axes[1][2].set_title('Metrics of RNN - Fasttext')
plt.subplots_adjust(hspace=0.5, wspace=0.4)
    
plt.show()



# In[6]:


import pandas as pd
import warnings
from performance_utils import calc_metrics, get_roc_auc
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from keras.optimizers import Adam
from keras.utils import set_random_seed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings('ignore')

ft = pd.read_csv('data/fasttext_100.csv')

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


# In[19]:


X = ft.drop(['overall', 'embeddings'], axis = 1)
y = ft['overall']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

y_pred = model.predict(X_test.drop(['reviewText','preprocessed_text'], axis = 1))
y_pred = (y_pred > 0.5).astype(int)

test_result = X_test[['reviewText', 'preprocessed_text','processed_len','review_length']]
test_result['actual'] = y_test
test_result['predicted'] = y_pred


# In[31]:


bad_predictions = test_result[test_result['actual'] != test_result['predicted']]
bad_predictions


# In[32]:


from wordcloud import WordCloud

fp = bad_predictions[bad_predictions['actual'] == 0]
print(fp.shape)
text_data = ' '.join(fp['preprocessed_text']) 
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_data)

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()


# - These are actually negative/bad review but we can clearly see words like good, better, great, best which indicate positivity.

# In[33]:


from wordcloud import WordCloud

fn = bad_predictions[bad_predictions['actual'] == 1]
print(fn.shape)
text_data = ' '.join(fn['preprocessed_text']) 
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_data)

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()


# -  This does not say a lot about why these are mislabeled as bad reviews.
# - Look more closely.

# In[37]:


sns.histplot(fn['processed_len'], alpha = 0.9)


# In[42]:


good_predictions = test_result[(test_result['actual'] == 0) | (test_result['actual'] == 1)]


# In[43]:


good_predictions[['review_length','processed_len']].describe()


# In[48]:


fp[['review_length','processed_len']].describe()


# In[49]:


fn[['review_length','processed_len']].describe()


#  - We can see that the processed length of reviews is highest for false negatives compared to false positives and good predictions. 
# 
# - We might want to set a cap to consider only first 'n' words of the review and retry to mitigate this problem.

# In[53]:


fn['reviewText'].tolist()[:4]


# - There might not be enough evidence to say why these good reviews have been identified as bad reviews.

# In[ ]:




