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
# Import for pre-processing, cross-validation, and model selection.
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone

MODEL_TO_CLASS_TO_DEFAULT_PARAMETERS = {
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
		"random_state": None,
		"verbose": 3
	}),
	"SKLsvm": (SVC, {
		"C": 1.0,
		"kernel": "rbf",
		"degree": 3,
		"gamma": "scale",
		"verbose": 3,
		"max_iter": -1,
		"random_state": None
	}),
	"SKLlogreg": (LogisticRegression, {
		"penalty": "l2",
		"tol": 0.0001,
		"C": 1.0,
		"random_state": None,
		"max_iter": 100,
		"verbose": 3
	}),
	"SKLhgb": (HistGradientBoostingClassifier, {
		"learning_rate": 0.1,
		"max_iter": 100,
		"max_leaf_nodes": 31,
		"max_depth": None,
		"min_samples_leaf": 20,
		"l2_regularization": 0.0,
		"verbose": 3,
		"random_state": None
	}),
	"XGBgb": (XGBClassifier, {
		"n_estimators": 100,
		"max_depth": 6,
		"learning_rate": 0.3,
		"verbosity": 3,
		"objective": "binary:logistic",
		"booster": "gbtree",
		"subsample": 1.0,
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
	"SKLstack": (StackingClassifier, {
		"estimators": [
			("SKLhgb", HistGradientBoostingClassifier(
				learning_rate=0.08564,
				max_iter=1, # 479
				max_leaf_nodes=75,
				max_depth=81,
				min_samples_leaf=15,
				random_state=7,
			)),
			("XGBgb", XGBClassifier(
				n_estimators=2, # 800
				max_depth=4, # 16
				learning_rate=0.3, # 0.1
				subsample=1, # 0.8
				reg_alpha=0.1,
				reg_lambda=2
			)),
			("CBgb", CatBoostClassifier(
				learning_rate=0.3, # 0.1
				max_depth=6, # 12
				n_estimators=2, # 800
				reg_lambda=None
			)),
			("SKLlogreg", LogisticRegression(
				penalty="l2",
				C=5.6,
				max_iter=400
			)),
			("SKLknn", KNeighborsClassifier(
				n_neighbors=3,
				weights="distance"
			))
		],
		"final_estimator": LogisticRegression(),
		"cv": 5
	})
}

S_train = pd.read_csv("./data/train.csv")
S_train_tfidf = pd.read_csv("./data/train_tfidf_features.csv")
S_test = pd.read_csv("./data/test.csv")
S_test_tfidf = pd.read_csv("./data/test_tfidf_features.csv")

X_train = S_train_tfidf.iloc[:, 2:].values
y_train = S_train["label"].values.reshape(-1, 1)
X_test = S_test_tfidf.iloc[:, 1:].values

def train_model(model_type, **kwargs):
	model_class, default_params = MODEL_TO_CLASS_TO_DEFAULT_PARAMETERS[model_type]
	params = {**default_params, **kwargs}
	model = model_class(**params)
	model.fit(X_train, y_train)
	return model

def predict_model(model, X): return model.predict(X)

def generate_predictions(model_type, **kwargs):
	start_time = time.time()
	model_class, default_params = MODEL_TO_CLASS_TO_DEFAULT_PARAMETERS[model_type]
	params = {**default_params, **kwargs}

	if model_type == "SKLstack":
		estimators = params.pop("estimators", default_params["estimators"])
		final_estimator_params = {
			key: params.pop(key) for key in kwargs if key in LogisticRegression().get_params()
		}
		final_estimator = params.pop("final_estimator", default_params["final_estimator"])
		if isinstance(final_estimator, LogisticRegression): final_estimator.set_params(**final_estimator_params)
		model = model_class(estimators=estimators, final_estimator=final_estimator, **params)
	else:
		model = model_class(**params)

	model.fit(X_train, y_train)
	predictions = predict_model(model, X_test)
	output_dir = f"./predictions/{model_type}/"
	os.makedirs(output_dir, exist_ok=True)
	file_name = os.path.join(output_dir, "predictions.csv")
	pd.DataFrame({"id": S_test["id"], "label": predictions}).to_csv(file_name, index=False)
	end_time = time.time()
	logging.info(f"Predictions file {file_name} generated in {end_time - start_time:.2f}s.")
	print(f"Predictions file {file_name} generated in {end_time - start_time:.2f}s.")