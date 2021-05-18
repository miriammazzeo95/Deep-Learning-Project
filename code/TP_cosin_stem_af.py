import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

filename = "code/data/new_TA_stem_af.csv"
df = pd.read_csv(
    filename,
    sep = ',',
    dtype = {"id": np.int32,
            "titolo": str,
            "utente": str,
            "domanda": str,
            "utente2": str,
            "utile2": bool,
            "risposta2": str,
            "utente3": str,
            "utile3": bool,
            "risposta3": str,
            "utente4": str,
            "utile4": bool,
            "risposta4": str,
            "utente5": str,
            "utile5": bool,
            "risposta5": str,
            "utente6": str,
            "utile6": bool,
            "risposta6": str,
            "utente7": str,
            "utile7": bool,
            "risposta7": str,
            "utente8": str,
            "utile8": bool,
            "risposta8": str,
            "utente9": str,
            "utile9": bool,
            "risposta9": str,
            "utente10": str,
            "utile10": bool,
            "risposta10": str},
    na_filter = False)

documents = df['domanda']

tfidf_vectorizer = TfidfVectorizer(use_idf=True) #  stop_words = stopwords
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

cs = cosine_similarity(tfidf_matrix, tfidf_matrix)
for i in range(0, cs.shape[0]):
    cs[i][i] = 0

count = 0
similarities = []
for i in range(0, cs.shape[0]):
    for j in range(0, cs.shape[1]):
        if(cs[i][j] > 0.99):
            if(i in similarities):
                continue
            similarities.append(j)
            print("Indexes " + str(i) + " - " + str(j) + " with Cosine " + str(cs[i][j]))
            print("Titolo 1: " + df['titolo'][i] + " e titolo 2: " + df['titolo'][j])
            count+=1
print(count)

print(df.shape)
for i in similarities:
    df = df.drop(df.index[i])
print(df.shape)

export_csv = df.to_csv(r'code/data/new_TA_stem_cosin_af_similarities.csv', index = None, header=True)