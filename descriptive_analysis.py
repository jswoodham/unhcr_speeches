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

# +
## Load up data

df = pd.read_feather("data/cleaned_speeches")

# +
## Bring in parameters

with open("params.yaml", "r") as fd:
    params = yaml.safe_load(fd)
    
punctuation = params['preprocessing']['punctuation'] 
stopwords = params['preprocessing']['stopwords'] + nltk.corpus.stopwords.words('english')

# +
## Tokenize and explode

df['speech'] = df.speech.apply(nltk.tokenize.word_tokenize)

df = df.explode('speech').reset_index()

# +
## Remove stopwords (to capture compounds like the United Nations)

df = df.filter_column_isin('speech',
                          stopwords,
                          complement = True)

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

df.to_feather(r'data/descriptive')
