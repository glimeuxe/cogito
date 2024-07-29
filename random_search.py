from base import *

def cv_random_search(model_type, param_distributions, k, n_iter=100):
	model_class, default_params = MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP[model_type]
	model = model_class(**default_params)
	random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=n_iter, cv=k, scoring="f1", random_state=42)
	random_search.fit(X_train, y_train)
	best_model, best_hyperparameters = random_search.best_estimator_, random_search.best_params_
	print("Best model:", best_model)
	print("Best hyperparameters:", best_hyperparameters)

	best_index = random_search.best_index_
	best_f1_score = random_search.cv_results_['mean_test_score'][best_index]
	print("Mean cross-validated f1 for best model:", best_f1_score)

	for i in range(len(random_search.cv_results_['params'])):
		print(f"Model {i + 1}:")
		print("Hyperparameters:", random_search.cv_results_['params'][i])
		print("Mean cross-validated f1:", random_search.cv_results_['mean_test_score'][i])
		print()
	# return best_model, best_hyperparameters