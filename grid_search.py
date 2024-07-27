from base import *

def cv_grid_search(model_type, grid, k):
	model_class, default_params = MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP[model_type]
	model = model_class(**default_params)
	grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=k, scoring="f1")
	grid_search.fit(X_train, y_train)
	best_model, best_hyperparameters = grid_search.best_estimator_, grid_search.best_params_
	print("Best model:", best_model)
	print("Best hyperparameters:", best_hyperparameters)

	best_index = grid_search.best_index_
	best_f1_score = grid_search.cv_results_['mean_test_score'][best_index]
	print("Mean cross-validated f1 for best model:", best_f1_score)

	for i in range(len(grid_search.cv_results_['params'])):
		print(f"Model {i + 1}:")
		print("Hyperparameters:", grid_search.cv_results_['params'][i])
		print("Mean cross-validated f1:", grid_search.cv_results_['mean_test_score'][i])
		print()
	# return best_model, best_hyperparameters