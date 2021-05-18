import numpy as np
import pandas as pd
import unidecode
import re
import string

import nltk
from nltk.corpus import stopwords
#You are currently trying to download every item in nltk data, so this can take long. You can try downloading only the stopwords that you need:

nltk.download('stopwords')
from nltk.stem.snowball import ItalianStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

filename = "df_politic.csv"
df = pd.read_csv(
  filename,
  sep = ',',
  usecols=["text"],
  na_filter = False)

# df_politic = pd.read_csv(
#   'df_politic.txt')

def tokenize(cell):
  regex_str = [
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',   # numbers
    r"(?:[a-z][a-z\-_]+[a-z])",     # words with -
    r'(?:[\w_]+)',                  # other words
    r'(?:\S)'                       # anything else
  ]
  cell = unidecode.unidecode(cell)
  tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
  return tokens_re.findall(cell.lower())
    
def remove_stop_words(cell):
  stop_words = stopwords.words('italian')
  return [word for word in cell if word not in stop_words]

def remove_punctation(cell):
  punctuation = string.punctuation
  return [word for word in cell if word not in punctuation]

def get_stemmed_text(cell):
  stemmer = ItalianStemmer()
  return [stemmer.stem(word) for word in cell]

def clean(col):
  col = col.apply(lambda x: tokenize(x))
  col = col.apply(lambda x: remove_stop_words(x))
  col = col.apply(lambda x: remove_punctation(x))
  col = col.apply(lambda x: get_stemmed_text(x))
  col = col.apply(lambda x: ' '.join(x))
  return col

def cleanAll(df):
  df['text'] = clean(df['text'])

  return df

df = cleanAll(df)

export_csv = df.to_csv(r'code/data/actual_similarq_answ_only_stem_mi_af.csv', index = None, header=True)

filename = "code/data/actual_similarq_answ_only_stem_mi_af.csv"
df = pd.read_csv(
  filename,
  sep = ',',
  na_filter = False)

for index, rows in df.iterrows():
  doc = [rows.mi, rows.af1, rows.af2, rows.af3, rows.af4, rows.af5, rows.af6, rows.af7, rows.af8, rows.af9]
  tfidf_vectorizer = TfidfVectorizer(use_idf=True)
  tfidf_matrix = tfidf_vectorizer.fit_transform(doc)

  cs = cosine_similarity(tfidf_matrix, tfidf_matrix)
  for i in range(0, cs.shape[0]):
      cs[i][i] = 0

  count = 0
  similarities = []
  i = 0
  threshold = 0.1
  for j in range(0, cs.shape[1]):
    if(cs[i][j] > threshold):
      if(i in similarities):
        continue
      similarities.append(j)
      print("Questions " + str(rows.id) + ": i " + str(i) + " - j " + str(j) + " with Cosine " + str(cs[i][j]))
      count+=1
  print("Questions ",str(rows.id)," had ",count," similar answers\n")




