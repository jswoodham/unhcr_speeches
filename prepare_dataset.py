# -*- coding: utf-8 -*-
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
#     display_name: Python [conda env:text-data-class]
#     language: python
#     name: conda-env-text-data-class-py
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

df = (pd.read_json("data/speeches.json"))
df = df[['author', 'content', 'id']]

# +
## Bring in parameters

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)
    
punctuation = params['preprocessing']['punctuation'] 
stopwords = params['preprocessing']['stopwords'] + nltk.corpus.stopwords.words('english')

# +
## Extract titles from content - text up until the first date is mentioned.

df['Title'] = df['content'].str.extract(r"\n+([\s\S]+), \d+\s\w+\s\d+")

# +
### Extract speeches and dates from content

## Speeches follow "Statements by the High Commissioner" and the date
date_text = (df['content'].
             str.extract(r"Statements by High Commissioner,\s*(\d+\s\w+\s\d+)[\s\n\r]+([\s\S]+)")
            )
date_text.columns = ['date', 'speech']
# -

date_text

# +
## Convert string dates to datetime
df['date'] = pd.to_datetime(date_text.date)

## Replace line breaks, double spaces with spaces
df['speech'] = date_text.speech.replace(to_replace = ['\n', '  '], value = ' ', regex = True)

## Lower case
df['speech'] = df.speech.str.lower()

## Remove dates and pesky punctuation
df['speech'] = df.speech.replace(to_replace = [r'(\d+\s\w+\s\d+)', r'([\'\"–]+)'], value = '', regex = True)

# +
## Superficial capitalization switches for my own satisfaction

df['author'] = (df.author.replace
                (to_replace = ['ogata', 
                               'guterres', 
                               'lubbers', 
                               'khan', 
                               'hartling', 
                               'schnyder', 
                               'lindt', 
                               'hocké', 
                               'stoltenberg'],
                 value      = ['Ogata', 
                               'Guterres', 
                               'Lubbers', 
                               'Khan', 
                               'Hartling', 
                               'Schnyder', 
                               'Lindt', 
                               'Hocké', 
                               'Stoltenberg'])
               )

df = df.rename(columns={'Title' : 'title', 'author' : 'speaker'})

# +
# Attempt to filter out Spanish and French (drops 10 speeches)

df = df[df["speech"].str.contains("acnur | réfugiés")==False].reset_index()
# -

## Add decade
df['decade'] = (df.date.dt.year//10)*10

# +
## Tokenize and explode

df['speech'] = df.speech.apply(nltk.tokenize.word_tokenize)

df = df.explode('speech').reset_index()

# +
## Lemmatize! We don't need tense, focus is on the topics we're covering, 
## and this way we'll reduce the likliehood that we're conflating meanings.

wnl = WordNetLemmatizer()

df['speech'] = ' '.join([wnl.lemmatize(w) for w in df.speech]).split()

# +
## Remove stopwords and punctuation

df = df.filter_column_isin('speech',
                          stopwords,
                          complement = True)

df = df.filter_column_isin('speech',
                          punctuation,
                          complement = True)

df.speech.value_counts().head(30)

# +
df = df[['id', 'speaker', 'date', 'title', 'speech', 'decade']].reset_index(drop = True)

df.to_feather(r'data/cleaned_speeches')
