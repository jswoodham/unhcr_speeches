# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python [conda env:unhcr_speeches]
#     language: python
#     name: conda-env-unhcr_speeches-py
# ---

import pandas as pd
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import yaml
import janitor as pj
import matplotlib.pyplot as plt
import numpy as np
import gensim
from gensim import corpora, models
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases, LdaModel
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pickle

# +
# Pull in model

topic_model = gensim.models.ldamodel.LdaModel.load('lda_model')
dictionary = corpora.Dictionary.load('speech.dict')
corpus = corpora.MmCorpus('speech.mm')
with open('data/docs_tagged', 'rb') as fp:
    docs_tagged = pickle.load(fp)

# +
# Generate dataframe of all document topics

topics = pd.DataFrame()

topics['topics'] = topic_model.get_document_topics(corpus)

sf = pd.DataFrame(data = topics.topics)

af = pd.DataFrame()

for i in range(10):
    af[str(i)]=[]

frames = [sf, af]
af = pd.concat(frames).fillna(0)

for i in range(693):
    for j in range(len(topics['topics'][i])):
        af[str(topics['topics'][i][j][0])].loc[i] = topics['topics'][i][j][1]
        
af = af.reset_index() 

## will merge on index - documents are in the same order as our tagged docs dataset, 
## which we can use the tags in to merge to the original dataset with date information

### https://stackoverflow.com/questions/66403628/how-to-change-topic-list-from-gensim-lda-get-document-topics-to-a-dataframe

# +
# Pull in tagged documents to merge tags to the document topic dataset

docs_tagged = pd.DataFrame(docs_tagged).explode('tags').reset_index()

df_topics = (
    af.merge(docs_tagged, on = 'index')
             [['tags', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            ).reset_index()

# +
# Pull in original dataset with date information

df_merge = (
    pd.read_feather('data/cleaned_speeches')
      [['id', 'date']]
      .drop_duplicates()
      .reset_index().reset_index()
    [['id', 'date']]
)

# +
# Merge topics dataframe to original dataframe 

df = (df_merge.merge(df_topics, left_on = 'id', right_on = 'tags')
     [['date', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']])

# Add a year variable
df['year'] = df['date'].dt.to_period('Y')

# +
# Rename and filter columns

df.columns = ['date',
              'topic 1', 
              'topic 2', 
              'topic 3', 
              'topic 4', 
              'topic 5', 
              'topic 6', 
              'topic 7', 
              'topic 8', 
              'topic 9', 
              'topic 10',
              'year']

df = df[['date',
        'year',
        'topic 1', 
        'topic 2', 
        'topic 3', 
        'topic 4', 
        'topic 5', 
        'topic 6', 
        'topic 7', 
        'topic 8', 
        'topic 9', 
        'topic 10']]

# +
# Prepare count variable and calculate relative topic frequency

df['count'] = 1

# +
# Sum topics by year

sums = (df
        .groupby('year')
        [['count', 'topic 1','topic 2','topic 3','topic 4','topic 5'
          ,'topic 6','topic 7','topic 8','topic 9','topic 10']]
        .sum()
       ) 

# +
# Divide topics by year by documents per year

topics_over_time = (sums[['topic 1','topic 2','topic 3','topic 4','topic 5',
                          'topic 6','topic 7','topic 8','topic 9','topic 10']]
                    .div(sums['count'], 
                         axis='index')
                   ).reset_index()
# -

topics_over_time

# +
# Save

topics_over_time.to_feather(r'data/topics_over_time')
