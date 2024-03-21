# Amazon-Sentiment-Analysis

## Overview

<p>Patient Notes Clustering project is about using the text written by numerous Physicians about 10 different patients. I turned this into an unsupervised machine learning by only considering the physician notes. This natural language processing (NLP) endeavor aims to uncover meaningful patterns and groupings within a dataset of physician notes. By leveraging advanced NLP techniques, we can gain valuable insights into the underlying structure of the data, potentially revealing trends and associations.</p>

## About Dataset

<p>The dataset was given as a part of my NLP coursework. The data has 3 columns out of which two prominent features are `case_num` (indicating patient - has 10 unique values representing 10 different patients) and `pn_history` (notes written by physician).</p>

<p>The `case_num` is used to compare the clustering plot results. On the `pn_history`, we apply Natural language processing techniques and Unsupervised Machine learning techniques to cluster the `pn_history`.</p>

## IDE and Environment

- Programming Language : Python
- IDE : Jupyter Notebooks
- Environment : environment.yml file included

## Data Cleaning and Data preprocessing

<p>There are no missing values in the data. As a part of data preprocessing, I first defined list of contractions and medical abbreviations to be replaced. Later, using regular expressions, I have converted the notes text to lower case, removed number, punctuations and special characters and stop words.</p>
<p>To reduce word to its root form, I compared the results of Lemmatizer and Stemming and decided to use Lemmatization. All these processes are implemented through a function called `preprocess_text`. Later, I have created two new columns, one for original note length and the other for processed text length for visualizations.</p>

## Visualizations

- The following histogram shows us the distribution of note length before and after preprocessing.

<img src="figs/document_length_frequency.png">

- Since we have 10 unique patients, I made the following plot to show the most frequent words in their combined physician notes. We can see that words like `year` are frequent in all the cases.

<img src="figs/common_words_per_patient_case.png">

- The following is a bar plot of top 20 words in all the notes combined irrespective of patient followed

<img src="figs/common_words_all_notes.png">  

## Using TFIDF

<p>I have defined my X (notes) and y (case_num - only to be used for verification). Using my X, I have build a Document Term Matrix (DTM using TFIDF vectorizor. However, instead of using every word occurance, I have filtered the words (which are features in DTM) using `min_df` and `max_df` parameters to remove most and least frequent words. The resulting DTM has 2478 features (prominent words).</p>

## Dimensionality Reduction

To tackle the curse of dimensionality are make it visualisable, I have tried different dimensionality reduction Techniques like `TruncatedSVD`, `UMAP` and `T-SNE`. The results of UMAP and T-SNE can easily be visualized in 2-Dimensions. 

- UMAP (2 Components):

<img src="figs/umap_2_component.png"> 

- T-SNE (2 Components):

<img src="figs/tsne_2components.png"> 


## Clustering

- Need to complete

## Results

- Need to complete




