import pandas as pd
import re
import json
import nltk
from nltk.corpus import stopwords
from mpstemmer import MPStemmer

def cleaning_data(df: pd.DataFrame, subset: list) -> pd.DataFrame:
  # 1. drop missing value
  df = df.dropna(ignore_index=True)
  # 2. drop duplicated
  df = df.drop_duplicates(subset=subset, ignore_index=True)
  return df

def case_folding(text: str) -> str:
  return text.lower()

def normalization(text: str, remove_punctuation_number: bool = True) -> str:
  # 1. remove urls, hashtags, and mentions
  text = re.sub(r'http[s]?://[\S]+', '', text)
  text = re.sub(r'[\S]+[\.|\s]com', '', text)
  text = re.sub(r'#[\w]+', '', text)
  text = re.sub(r'@[\w]+', '', text)
  text = re.sub(r'RT[\s]+', '', text)
  # 2. remove punctuation
  if remove_punctuation_number:
    text = re.sub(r'\&[\#]?[\w]+\;', '', text)
    text = re.sub(r'[\_|\'|\.|\,]', '', text)
    text = re.sub(r'[\-]', ' ', text)
    text = re.sub(r'[\W]', ' ', text)
  # 3. convert slang words
  normals = dict()
  try:
    with open('Data/2_Preprocessing_Data/slang_words.json', 'r') as file: # from dataset
      normals.update(json.load(file)) # slang word dari library
    with open('Data/2_Preprocessing_Data/slang_words_manual.json', 'r') as file: # from manual
      normals.update(json.load(file)) # slang word dari manual/perbaiki sendiri
  except FileNotFoundError as e:
    with open('../Data/2_Preprocessing_Data/slang_words.json', 'r') as file: # from dataset
      normals.update(json.load(file)) # slang word dari library
    with open('../Data/2_Preprocessing_Data/slang_words_manual.json', 'r') as file: # from manual
      normals.update(json.load(file)) # slang word dari manual/perbaiki sendiri
  text = f' {text} ' # implement
  for i in normals:
    text = text.replace(i, normals[i])
  # 4. remove number
  if remove_punctuation_number:
    text = re.sub(r'\d+', '', text)
  return text

def tokenizing(text: str) -> list:
  return text.split()

def stopword_removal(text: list) -> list:
  nltk.download('stopwords')
  stop_words = set(stopwords.words('indonesian'))
  try:
    with open('Data/2_Preprocessing_Data/stopwords.txt', 'r') as file:
      more_stopwords = file.read().split('\n')
  except FileNotFoundError as e:
    with open('../Data/2_Preprocessing_Data/stopwords.txt', 'r') as file:
      more_stopwords = file.read().split('\n')
  stop_words.update(more_stopwords)
  return [word for word in text if word not in stop_words]

def stemming(text: list) -> list:
  stemmer = MPStemmer()
  return [stemmer.stem_kalimat(word) for word in text]

def remove_outlier(df: pd.DataFrame) -> pd.DataFrame:
  Q1 = df['word_count'].quantile(0.25)
  Q3 = df['word_count'].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df = df[(df['word_count'] >= lower_bound) & (df['word_count'] <= upper_bound)]
  return df