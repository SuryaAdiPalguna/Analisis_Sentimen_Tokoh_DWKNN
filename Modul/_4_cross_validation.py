import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator
from Modul._5_implementasi_algoritma import DWKNN

def k_fold_cross_validation(n: int, model: BaseEstimator, X: np.array, y: np.array):
  model = DWKNN(k=5)
  kf = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
  scores = cross_val_score(model, X, y, cv=kf)
  return scores