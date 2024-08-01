from base import *

def cv_random_search(model_type, grid, k, n_iter=2):
	model_class, default_params = MODEL_TO_CLASS_TO_DEFAULT_PARAMETERS[model_type]
	if model_type.startswith("SKLstack"):
		grid = {
			("final_estimator__" + k if k in LogisticRegression().get_params() else k): v
			for k, v in grid.items()
		}

	model = model_class(**default_params)
	random_search = RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=n_iter, cv=k, scoring="f1", verbose=3)
	random_search.fit(X_train, y_train.ravel())

	best_model = random_search.best_estimator_
	best_hyperparameters = random_search.best_params_
	best_index = random_search.best_index_
	best_f1 = random_search.cv_results_["mean_test_score"][best_index]

	print("Best hyperparameters:", best_hyperparameters)
	print("Mean cross-validated f1 for best model:", best_f1)
	logging.info("Best hyperparameters: %s", best_hyperparameters)
	logging.info("Mean cross-validated f1 for best model: %.4f", best_f1)
	for i in range(len(random_search.cv_results_["params"])):
		logging.info(f"Model {i + 1}:")
		logging.info("Hyperparameters: %s", random_search.cv_results_["params"][i])
		logging.info("Mean cross-validated f1: %.4f", random_search.cv_results_["mean_test_score"][i])
	return best_model, best_hyperparameters