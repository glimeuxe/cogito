from base import *

def cv_grid_search(model_type, grid):
	model_class, default_params = MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP[model_type]
	model = model_class(**default_params)
	gridsearch = GridSearchCV(estimator=model, param_grid=grid, cv=5, scoring="accuracy")
	gridsearch.fit(X_train, y_train)
	best_model, best_hyperparameters = gridsearch.best_estimator_, gridsearch.best_params_
	print("Best model:", best_model)
	print("Best hyperparameters:", best_hyperparameters)
	# return best_model, best_hyperparameters