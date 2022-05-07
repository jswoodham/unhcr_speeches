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

df = (pd.read_json("UNHCR speeches/speeches.json"))
df = df[['author', 'content']]

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

# +
## Convert string dates to datetime
df['date'] = pd.to_datetime(date_text.date)

## Replace line breaks, double spaces with spaces
df['speech'] = date_text.speech.replace(to_replace = ['\n', '  '], value = ' ', regex = True)

## Remove dates
df['speech'] = df.speech.replace(to_replace = r"(\d+\s\w+\s\d+)", value = '', regex = True)

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

df = df.rename(columns={'Title' : 'title'})
# -

## Final dataset
df = df[['author', 'date', 'title', 'speech']]
df.sample(10)

df.to_json(r'data/cleaned_speeches.json')
