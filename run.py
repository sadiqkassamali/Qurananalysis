

# Turns on/off pretty printing
from pprint import pprint


# Every returned Out[] is displayed, not just the last one.
import nltk
import null as null
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity =  "all"

# Load CSV (using python)
import csv
import numpy as nu
import pandas as pd
import gensim
#import scikit
import wordcloud
import matplotlib
import gensim
from gensim.models import Word2Vec
from sklearn.utils.tests.test_pprint import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from nltk import word_tokenize
nltk.download('punkt')

d=pd.read_csv('data\quran.csv')
if (len(d) > 0) :
    pprint("print total rows count: ")
    pprint(len(d))
    #print("print total data before NaN removal: ")
    #print(d)
    d.fillna('', inplace=True)
    # print("print total data after NaN removal: " )
    #print(d)
    dropped_column = d.drop(d.columns[-1],axis=1)  # Drop last column
    pprint("print total data After last column removal: ")
    pprint(dropped_column)
    #print(dropped_column)
    #print(dropped_column.head()) #if you want to view first n-rows
    #print(dropped_column.tail())  # if you want to view last  n-rows
    #print(dropped_column[20:30 + 1])  #slice and dice of rows from 20 : 30
    #print(dropped_column.describe()) #Statistics of your dataframe if value is int

#spliting each word
if (len(dropped_column['Text']) > 0) :
    dropped_column['Text'] = dropped_column['Text'].str.split()
pprint("Splitting each word in Text : ")
pprint(dropped_column)

# You can filter for one surah too if you want!
verses = dropped_column['Text'].values.tolist()
pprint(verses)

#tokenization of text

def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return word_tokenize(text)
    verses = verses.column.apply(custom_tokenize)

    sentences = [word_tokenize(s) for s in verses]
    #printing tokens
    pprint(sentences)

