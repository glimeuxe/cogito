import logging, os, time, glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone

TOP_ESTIMATOR_1 = ("SKLlogreg", LogisticRegression(
	C=0.64,
	class_weight="balanced"
))

TOP_ESTIMATOR_2 = ("CBgb", CatBoostClassifier(
	iterations=900,
	learning_rate=0.1,
	depth=6,
	rsm=0.8,
	auto_class_weights="Balanced",
	random_strength=1.2,
	bootstrap_type="MVS",
	subsample=0.8
))

TOP_ESTIMATOR_3 = ("SKLet", ExtraTreesClassifier(
	n_estimators=150,
	criterion="gini",
	max_depth=600,
	min_samples_split=160,
	min_impurity_decrease=0.00001,
	bootstrap=True,
	class_weight="balanced_subsample",
	ccp_alpha=0.00001,
	max_samples=0.9
))

TOP_ESTIMATOR_4 = ("SKLmnb", MultinomialNB(
	alpha=1.38,
	fit_prior=False
))

TOP_ESTIMATOR_5 = ("SKLlsvm", LinearSVC(
	penalty="l2",
	loss="hinge",
	dual=True,
	C=0.3425,
	class_weight="balanced",
	max_iter=2000
))

MODEL_TO_CLASS_TO_DEFAULT_PARAMETERS = {
	"SKLet": (ExtraTreesClassifier, {
		"n_estimators": 100,
		"criterion": "gini",
		"max_depth": None,
		"min_samples_split": 2,
		"min_samples_leaf": 1,
		"min_weight_fraction_leaf": 0.0,
		"max_features": "sqrt",
		"max_leaf_nodes": None,
		"min_impurity_decrease": 0.0,
		"bootstrap": False,
		"random_state": None,
		"verbose": 0,
		"class_weight": None,
		"ccp_alpha": 0.0,
		"max_samples": None
	}),
	"SKLlsvm": (LinearSVC, {
		"penalty": "l2",
		"loss": "squared_hinge",
		"dual": "auto",
		"tol": 0.0001,
		"C": 1.0,
		"class_weight": None,
		"verbose": 0,
		"random_state": None,
		"max_iter": 1000
	}),
	"SKLmnb": (MultinomialNB, {
		"alpha": 1.0,
		"fit_prior": True
	}),
	"SKLlogreg": (LogisticRegression, {
		"penalty": "l2",
		"tol": 0.0001,
		"C": 1.0,
		"class_weight": None,
		"random_state": None,
		"max_iter": 100,
		"verbose": 0
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
		"max_leaves": 0,
		"learning_rate": 0.3,
		"verbosity": 1,
		"objective": "binary:logistic",
		"booster": "gbtree",
		"gamma": 0,
		"subsample": 1.0,
		"reg_alpha": 0,
		"reg_lambda": 1,
		"random_state": None
	}),
	"CBgb": (CatBoostClassifier, {
		"iterations": None,
		"learning_rate": None,
		"depth": None,
		"l2_leaf_reg": None,
		"rsm": None,
		"random_seed": None,
		"auto_class_weights": None,
		"random_strength": None,
		"boosting_type": "Plain",
		"bootstrap_type": None,
		"subsample": None,
		"grow_policy": None,
		"penalties_coefficient": None
	}),
	"SKLstack": (StackingClassifier, {
		"estimators": [TOP_ESTIMATOR_2, TOP_ESTIMATOR_3, TOP_ESTIMATOR_4],
		"final_estimator": LogisticRegression(),
		"cv": 5
	})
}

S_train = pd.read_csv("train.csv")
S_train_tfidf = pd.read_csv("train_tfidf_features.csv")
S_test = pd.read_csv("test.csv")
S_test_tfidf = pd.read_csv("test_tfidf_features.csv")

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

	if model_type.startswith("SKLstack"):
		estimators = params.pop("estimators", default_params["estimators"])
		final_estimator = params.pop("final_estimator", default_params["final_estimator"])
		final_estimator_params = {
			key: params.pop(key) for key in kwargs if key in LogisticRegression().get_params()
		}
		final_estimator.set_params(**final_estimator_params)
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