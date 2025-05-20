import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from Modul._2_preprocessing_data import cleaning_data, case_folding, normalization, tokenizing, stopword_removal, stemming, remove_outlier
from Modul._3_feature_extraction import tfidf
from Modul._5_implementasi_algoritma import DWKNN, oversampling

model = DWKNN(k=5)
X_train = None
y_train = None

def train_dwknn_model(df: pd.DataFrame):
	global X_train, y_train

	df['clean_full_text'] = df['full_text'].apply(lambda data: normalization(text=data, remove_punctuation_number=False))
	df['word_count'] = df["clean_full_text"].str.split().str.len()
	print("selesai 1")

	df = cleaning_data(df, subset=['full_text'])
	print("selesai 2")
	df['text_preprocessed'] = df['full_text'].apply(case_folding)
	print("selesai 2")
	df['text_preprocessed'] = df['text_preprocessed'].apply(normalization)
	print("selesai 2")
	df['text_preprocessed'] = df['text_preprocessed'].apply(tokenizing)
	print("selesai 2")
	df['text_preprocessed'] = df['text_preprocessed'].apply(stopword_removal)
	print("selesai 2")
	df['text_preprocessed'] = df['text_preprocessed'].apply(stemming)
	print("selesai 2")
	df['text_preprocessed'] = df['text_preprocessed'].apply(lambda text: ' '.join(text))
	df = remove_outlier(df)
	print("selesai 2")
	df.loc[df['sentiment'] == 'positif', 'sentiment'] = 1
	df.loc[df['sentiment'] == 'netral', 'sentiment'] = 0
	df.loc[df['sentiment'] == 'negatif', 'sentiment'] = -1
	print("selesai 2")

	df = df.replace("", np.nan)
	df = cleaning_data(df, subset=['text_preprocessed'])
	tfidf_df = tfidf(df)
	print("selesai 3")

	df['sentiment'] = df['sentiment'].astype(int)
	X_train = tfidf_df
	y_train = df['sentiment']
	X_train, y_train = oversampling(X_train, y_train)
	print("selesai 4")

	# return X_train, y_train

def predict_sentiment(text: str):
	global model, X_train, y_train

	print(X_train)
	print(y_train)

	df = pd.DataFrame({'full_text': [text]})

	df['text_preprocessed'] = df['full_text'].apply(case_folding)
	df['text_preprocessed'] = df['text_preprocessed'].apply(normalization)
	df['text_preprocessed'] = df['text_preprocessed'].apply(tokenizing)
	df['text_preprocessed'] = df['text_preprocessed'].apply(stopword_removal)
	df['text_preprocessed'] = df['text_preprocessed'].apply(stemming)
	df['text_preprocessed'] = df['text_preprocessed'].apply(lambda text: ' '.join(text))

	tfidf_df = tfidf(df)

	X_test = tfidf_df
	X = pd.concat([X_train, X_test], ignore_index=False).fillna(0).reset_index(drop=True)
	X_train = X.loc[:len(X_train)-1].reset_index(drop=True)
	X_test = X.loc[len(X_train):].reset_index(drop=True)
	X_train_np = np.array(X_train)
	X_test_np = np.array(X_test)
	y_train_np = np.array(y_train)
	model.fit(X_train_np, y_train_np)
	y_predict = model.predict(X_test_np)

	return y_predict[0]