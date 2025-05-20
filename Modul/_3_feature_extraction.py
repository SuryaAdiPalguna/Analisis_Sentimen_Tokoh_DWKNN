import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

def tfidf(df: pd.DataFrame) -> pd.DataFrame:
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(df['text_preprocessed'])
  features = vectorizer.get_feature_names_out()
  tftd = csr_matrix(X).toarray()
  tf = np.where(tftd > 0, 1 + np.log(tftd), 0)
  N = len(df)
  dft = np.array(tftd > 0).sum(axis=0)
  idf = np.log(N / dft)
  tfidf = tf * idf
  tfidf_df = pd.DataFrame(tfidf, columns=features, index=df.index)
  return tfidf_df