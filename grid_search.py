from base import *

def cv_grid_search(model_type, grid, k):
	model_class, default_params = MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP[model_type]
	model = model_class(**default_params)
	gridsearch = GridSearchCV(estimator=model, param_grid=grid, cv=k, scoring="accuracy")
	gridsearch.fit(X_train, y_train)
	best_model, best_hyperparameters = gridsearch.best_estimator_, gridsearch.best_params_
	print("Best model:", best_model)
	print("Best hyperparameters:", best_hyperparameters)

	best_index = gridsearch.best_index_
	best_f1_score = gridsearch.cv_results_['mean_test_score'][best_index]
	print("Mean cross-validated f1 for best model:", best_f1_score)

	for i in range(len(gridsearch.cv_results_['params'])):
		print(f"Model {i + 1}:")
		print("Hyperparameters:", gridsearch.cv_results_['params'][i])
		print("Mean cross-validated f1:", gridsearch.cv_results_['mean_test_score'][i])
		print()
	# return best_model, best_hyperparameters