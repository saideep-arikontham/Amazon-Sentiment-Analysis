#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import GridSearchCV
import warnings
from performance_utils import get_scores
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

wv = pd.read_csv('data/word2vec_100.csv')
sg = pd.read_csv('data/skipgram_100.csv')
ft = pd.read_csv('data/fasttext_100.csv')


# In[2]:


scores_data = []


# # Using XGBoost

# ### CBOW

# In[3]:


# Separate features and target
X = wv.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = wv['overall']

# Define the parameter grid for XGBClassifier
param_grid = {
    'xgb__max_depth': [6, 9],
    'xgb__max_delta_step': [4, 6],
    'xgb__subsample': [0.5],
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline with train_test_split and XGBClassifier
pipeline = Pipeline([
    #Dimension reduction step if we use it.
    ('xgb', XGBClassifier(random_state = 42, eval_metric='auc', scale_pos_weight = 0.15176234428197508))])

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_

print(best_params)
best_estimator = grid_search.best_estimator_

scores_data.append(['XGBoost', 'CBOW'] + get_scores(best_estimator, X_train, y_train, X_test, y_test, 'figs/xgb_cbow.png'))


# ### Skipgram

# In[4]:


# Separate features and target
X = sg.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = sg['overall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# Define the parameter grid for XGBClassifier
param_grid = {
    'xgb__max_depth': [6, 9],
    'xgb__max_delta_step': [4, 6],
    'xgb__subsample': [0.5]
}


# Create a pipeline with train_test_split and XGBClassifier
pipeline = Pipeline([
    #Dimension reduction step if we use it.
    ('xgb', XGBClassifier(random_state = 42, eval_metric='auc', scale_pos_weight = 0.15176234428197508))])

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_

print(best_params)
best_estimator = grid_search.best_estimator_

scores_data.append(['XGBoost', 'Skipgram'] + get_scores(best_estimator, X_train, y_train, X_test, y_test, 'figs/xgb_skipgram.png'))


# ### Fasttext

# In[5]:


# Separate features and target
X = ft.drop(['overall','reviewText', 'preprocessed_text', 'embeddings'], axis = 1)
y = ft['overall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# Define the parameter grid for XGBClassifier
param_grid = {
    'xgb__max_depth': [6, 9],
    'xgb__max_delta_step': [4, 6],
    'xgb__subsample': [0.5]
}


# Create a pipeline with train_test_split and XGBClassifier
pipeline = Pipeline([
    #Dimension reduction step if we use it.
    ('xgb', XGBClassifier(random_state = 42, eval_metric='auc', scale_pos_weight = 0.15176234428197508))])

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_

print(best_params)
best_estimator = grid_search.best_estimator_

scores_data.append(['XGBoost', 'Fasttext'] + get_scores(best_estimator, X_train, y_train, X_test, y_test, 'figs/xgb_fasttext.png'))


# In[6]:


scores = pd.DataFrame(data = scores_data, columns = ['model', 'embedding','accuracy','precision','recall','f1 score','roc auc'])
scores


# In[7]:


scores.to_csv('data/XGBoost_performance.csv', index = False)


# In[ ]:




