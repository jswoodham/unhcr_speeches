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
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import yaml
import janitor as pj
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
from hdbscan import HDBSCAN
import gensim
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases, LdaModel
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os
import logging
import pickle


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

df = pd.read_feather('data/cleaned_speeches')

# +
with open('params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

d_f = params['preprocessing']['df']
stopwords = params['preprocessing']['stopwords'] + nltk.corpus.stopwords.words('english') +nltk.corpus.stopwords.words('spanish') + nltk.corpus.stopwords.words('french')
punctuation = params['preprocessing']['punctuation']
passes = params['lda']['passes']
iterations = params['lda']['iterations']
num_topics = params['lda']['num_topics']

# +
# Tokenize

df['speech'] = df.speech.apply(nltk.tokenize.word_tokenize) 
# -

df = df.explode('speech').reset_index()

# +
# Lemmatize

wnl = WordNetLemmatizer()

df['speech'] = ' '.join([wnl.lemmatize(w) for w in df.speech]).split()

# +
# Remove stopwords and punctuation

df = df.filter_column_isin('speech',
                          stopwords,
                          complement = True)

df = df.filter_column_isin('speech',
                          punctuation,
                          complement = True)
# -

df = df.groupby(['id', 'speaker', 'date', 'title', 'decade'])['speech'].apply(' '.join).reset_index()

# +
# Re-tokenize 

df['speech'] = df.speech.apply(nltk.tokenize.word_tokenize) 

# +
# Make the text for each document a list of tokens for bigrams/LDA

docs_tagged = (
    df
    .apply(lambda row: TaggedDocument(row.speech, [row.id]), axis = 1)
    .tolist()
)

# +
# Clean off the tags because they confuse the bigrams 

docs = pd.DataFrame(docs_tagged)

docs = docs['words'].tolist()

# +
bigram = Phrases(docs, min_count = 20)

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)


# +
# Dictionary and corpus function

def prep_corpus(docs, no_below=d_f['min'], no_above=d_f['max']):
    print('Building dictionary...')
    dictionary = Dictionary(docs)
#     print(dictionary)
    stopword_ids = map(dictionary.token2id.get, stopwords)
#     print(stopword_ids)
    dictionary.filter_tokens(stopword_ids)
#     print(dictionary)
    dictionary.compactify()
#     print(dictionary)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    print(dictionary)
    dictionary.compactify()
    
    print('Building corpus...')
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    return dictionary, corpus

### https://github.com/XuanX111/Friends_text_generator/blob/master/Friends_LDAvis_Xuan_Qi.ipynb


# -

dictionary, corpus = prep_corpus(docs)

MmCorpus.serialize('speech.mm', corpus)
dictionary.save('speech.dict')

lda_model = LdaModel(corpus=corpus,
         num_topics = num_topics,
         eval_every = 1,
         passes = passes,
         iterations = iterations,
         id2word=dictionary,
         random_state=np.random.RandomState(42))

lda_model.save('lda_model')


with open('data/docs', "wb") as fp:   #Pickling
    pickle.dump(docs, fp)

with open('data/docs_tagged', 'wb') as fp:
    pickle.dump(docs_tagged, fp)
