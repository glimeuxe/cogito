import logging, os, time
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Import for pre-processing.
from sklearn.preprocessing import StandardScaler
# Import for cross-validation and model selection.
from joblib import Parallel, delayed
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
# from sklearn.metrics import

MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP = {
	"SKLknn": (KNeighborsClassifier, {
		"n_neighbors": 5,
		"weights": "uniform",
		"p": 2,
		"metric": "minkowski",
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
		"random_state": None,
		"verbose": 1
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
	}),
	"SKLlogreg": (LogisticRegression, {
		"penalty": "l2",
		"C": 1.0,
		"solver": "lbfgs",
		"max_iter": 100,
		"random_state": None
	}),
	"SKLstack": (StackingClassifier, {
		"estimators": [
			("SKLhgb", HistGradientBoostingClassifier(
				learning_rate=0.1,
				max_iter=1,
			)),
			("XGBgb", XGBClassifier(
				n_estimators=1,
				max_depth=1,
				learning_rate=0.1,
				use_label_encoder=False,
				eval_metric="logloss"
			)),
			("CBgb", CatBoostClassifier(
				learning_rate=0.1,
				n_estimators=1,
				max_depth=1
			))
		],
		"final_estimator": LogisticRegression(),
		"cv": 2
	}),
}

S_train = pd.read_csv("./data/train.csv")
S_train_tfidf = pd.read_csv("./data/train_tfidf_features.csv")
S_test = pd.read_csv("./data/test.csv")
S_test_tfidf = pd.read_csv("./data/test_tfidf_features.csv")

X_train = S_train_tfidf.iloc[:, 2:].values
y_train = S_train["label"].values.reshape(-1, 1)
X_test = S_test_tfidf.iloc[:, 1:].values

def train_model(model_type, X_train, y_train, **kwargs):
	model_class, default_params = MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP[model_type]
	params = {**default_params, **kwargs}
	model = model_class(**params)
	model.fit(X_train, y_train)
	return model

def predict_model(model, X): return model.predict(X)

def generate_predictions(model_type, **kwargs):
	start_time = time.time()
	model_class, default_params = MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP[model_type]
	if model_type == "SKLstack":
		base_estimators = kwargs.pop("base_estimators", default_params["estimators"])
		final_estimator = kwargs.pop("final_estimator", default_params["final_estimator"])
		model = model_class(estimators=base_estimators, final_estimator=final_estimator, **kwargs)
	else:
		model = train_model(model_type, np.array(X_train), np.array(y_train), **kwargs)

	model.fit(X_train, y_train.ravel())
	predictions = predict_model(model, np.array(X_test))
	output_dir = f"./predictions/{model_type}/"
	os.makedirs(output_dir, exist_ok=True)
	file_name = os.path.join(output_dir, "predictions.csv")
	pd.DataFrame({"id": S_test["id"], "label": predictions}).to_csv(file_name, index=False)
	end_time = time.time()
	logging.info(f"Predictions file {file_name} generated in {end_time - start_time:.2f}s.")
	print(f"Predictions file {file_name} generated in {end_time - start_time:.2f}s.")