import pandas as pd
import numpy as np
import unidecode
import re
import string

#import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem.snowball import EnglishStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity

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
  stop_words = stopwords.words('english')
  return [word for word in cell if word not in stop_words]

def remove_punctation(cell):
  punctuation = string.punctuation
  return [word for word in cell if word not in punctuation]

def get_stemmed_text(cell):
  stemmer = EnglishStemmer()
  return [stemmer.stem(word) for word in cell]

def clean(col):
  col = col.apply(lambda x: tokenize(x))
  col = col.apply(lambda x: remove_stop_words(x))
  col = col.apply(lambda x: remove_punctation(x))
  col = col.apply(lambda x: get_stemmed_text(x))
  col = col.apply(lambda x: ' '.join(x))
  return col

def cleanAll(df):
  df = clean(df[df.columns[0]])
  return df


################## to run only once in order to create the txt files ##########################
#################################  first read files from csv ...slow ##########################
#### save text in a txt file
#### delete the file before rerunning, it overwrites!!
##### clean text and save cleaned text in a txt file

# filename = "Political-media-DFE.csv"
# df_politic = pd.read_csv(
#   filename,
#   sep = ',',
#   usecols=["text"],
#   na_filter = False,
#     encoding ="ISO-8859-1")

# filename = "export_dashboard_x_uk.xlsx"
# df_tweet_UK = pd.read_excel(
#   filename,
#   sheet_name="Stream",
#   usecols=[6],
#   na_filter = False)

# filename = "../data/dashboard_x_usa.xlsx"
# df_tweet_USA = pd.read_excel(
#   filename,
#   sheet_name="Stream",
#   usecols=[6],
#   na_filter = False, encoding ="ISO-8859-1")

### save text in a txt file
# df_politic.to_csv('../data/df_politic.txt',  index=None, sep=' ', mode='a')
# df_tweet_UK.to_csv('../data/df_tweet_uk.txt', index=None, sep=' ', mode='a')
# df_tweet_USA.to_csv('../data/df_tweet_usa.txt', index=None, sep=' ', mode='a')

###### clean text and save cleaned text in a txt file
# df_politic_c = cleanAll(df_politic)
# df_tweetUK_c = cleanAll(df_tweet_UK)
# df_tweetUSA_c = cleanAll(df_tweet_USA)

# df_politic_c.to_csv(r'df_politic_c.txt', index = None, sep=' ', mode='a')
# df_tweetUK_c.to_csv(r'df_tweetUK_c.txt', index = None, sep=' ', mode='a')
# df_tweetUSA_c.to_csv(r'../data/df_tweetUSA_c.txt', index = None, sep=' ', mode='a')

cleaner = 1

if cleaner == 0:
    ##Faster reading txt !!!
    df_politic = pd.read_csv(
      '../data/df_politic.txt')
    
    df_tweetUK = pd.read_csv(
      '../data/df_tweet_uk.txt', dtype=str, na_filter = False)
    df_tweetUK=df_tweetUK.rename(columns={'Tweet content': 'text'})
    
    df_tweetUSA = pd.read_csv(
      '../data/df_tweet_usa.txt', dtype=str, na_filter = False)
    df_tweetUSA=df_tweetUSA.rename(columns={'Tweet content': 'text'})
else:
    ##Faster reading txt !!!
    df_politic = pd.read_csv(
      '../data/df_politic_c.txt', dtype=str, na_filter = False)
    
    df_tweetUK = pd.read_csv(
      '../data/df_tweetUK_c.txt', dtype=str, na_filter = False)
    df_tweetUK = df_tweetUK.rename(columns={'Tweet content': 'text'})
    
    df_tweetUSA = pd.read_csv(
      '../data/df_tweetUSA_c.txt', dtype=str, na_filter = False)
    df_tweetUSA = df_tweetUSA.rename(columns={'Tweet content': 'text'})

#flattening nested list -> list of strings
df_politic_l = [item for items in df_politic.values.tolist() for item in items]
df_tweetUK_l = [item for items in df_tweetUK.values.tolist() for item in items]
df_tweetUSA_l = [item for items in df_tweetUSA.values.tolist() for item in items]



df=pd.concat([df_politic,df_tweetUK,df_tweetUSA])
df = [item for items in df.values.tolist() for item in items]

# df=df['text']

tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=0.0001)

vec = tfidf_vectorizer.fit(df)
tfidf_politic = vec.transform(df_politic_l)
tfidf_UK = vec.transform(df_tweetUK_l)
tfidf_USA = vec.transform(df_tweetUSA_l)

persentage_train=0.8

mP_train = tfidf_politic[0:int(np.ceil(5000*persentage_train))]
mP_test = tfidf_politic[int(np.ceil(5000*persentage_train)):]

mUK_train = tfidf_UK[0:int(np.ceil(169033*persentage_train))]
mUK_test = tfidf_UK[int(np.ceil(169033*persentage_train)):]

mUSA_train = tfidf_USA[0:int(np.ceil(204820*persentage_train))]
mUSA_test = tfidf_USA[int(np.ceil(204820*persentage_train)):]

####to inspect matrix
# print(tfidf_politic)

####to see dimension
# tfidf_politic
###<5000x7636 sparse matrix of type '<class 'numpy.float64'>' with 77048 stored elements in Compressed Sparse Row format>
    


