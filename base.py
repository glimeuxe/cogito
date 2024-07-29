import logging, os, time
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Import for pre-processing.
from sklearn.preprocessing import StandardScaler
# Import for cross-validation and model selection.
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
# from sklearn.metrics import make_scorer

MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP = {
	"SKLknn": (KNeighborsClassifier, {
		"n_neighbors": 5,
		"weights": "uniform",
		"p": 2,
		"metric": "minkowski"
	}),
	"SKLrf": (RandomForestClassifier, {
		"n_estimators": 100,
		"criterion": "gini",
		"max_depth": None,
		"min_samples_split": 2,
		"min_samples_leaf": 1,
		"max_leaf_nodes": None,
		"random_state": None
	}),
	"SKLsvm": (SVC, {
		"C": 1.0,
		"kernel": "rbf",
		"degree": 3,
		"gamma": "scale",
		"max_iter": -1,
		"random_state": None
	}),
	"SKLgb": (GradientBoostingClassifier, {
		"learning_rate": 0.1,
		"n_estimators": 100,
		"subsample": 1.0,
		"min_samples_split": 2,
		"min_samples_leaf": 1,
		"max_depth": 3,
		"random_state": None,
		"max_leaf_nodes": None
	}),
	"SKLhgb": (HistGradientBoostingClassifier, {
		"learning_rate": 0.1,
		"max_iter": 100,
		"max_leaf_nodes": 31,
		"max_depth": None,
		"min_samples_leaf": 20,
		"random_state": None
	}),
	"XGBgb": (XGBClassifier, {
		"n_estimators": 100,
		"max_depth": 3,
		"max_leaves": 0,
		"learning_rate": 0.1,
		"objective": "binary:logistic",
		"booster": "gbtree",
		"subsample": 1,
		"reg_alpha": 0,
		"reg_lambda": 1,
		"random_state": None
	}),
	"CBgb": (CatBoostClassifier, {
		"learning_rate": None,
		"subsample": None,
		"max_depth": None,
		"n_estimators": None,
		"reg_lambda": None,
		"random_state": None
	})
}

S_train = pd.read_csv("./data/train.csv")
S_train_tfidf = pd.read_csv("./data/train_tfidf_features.csv")
S_test = pd.read_csv("./data/test.csv")
S_test_tfidf = pd.read_csv("./data/test_tfidf_features.csv")

X_train = S_train_tfidf.iloc[:, 2:].values
y_train = S_train["label"].values.reshape(-1, 1)
X_test = S_test_tfidf.iloc[:, 1:].values