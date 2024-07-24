from base import *

class SimulatedAnnealing:
	def __init__(self, estimator, grid, scoring=f1_score, T=10, T_min=0.001, α=0.9, n_trans=5, max_iter=100, max_runtime=60, cv=5, n_jobs=-1, max_f1=np.inf, min_improvement=1e-4, patience=10):
		self.estimator = estimator
		self.grid = grid
		self.scoring = scoring
		self.T = T
		self.T_min = T_min
		self.α = α
		self.n_trans = n_trans
		self.max_iter = max_iter
		self.max_runtime = max_runtime
		self.cv = cv
		self.n_jobs = n_jobs
		self.max_f1 = max_f1
		self.min_improvement = min_improvement
		self.patience = patience

		self.best_hyperparameters_ = None
		self.best_f1_ = None
		self.grid_f1s_ = None
		self.runtime_ = None
		self._set_dynamic_params()

	def _set_dynamic_params(self):
		num_hyperparameters = np.prod([len(v) for v in self.grid.values() if isinstance(v, list)])
		self.T = 10 * num_hyperparameters
		self.T_min = 0.01 * self.T
		self.α = 0.9

	def _accept_prob(self, old_f1, new_f1, T):
		T += 0.01
		return np.exp((new_f1 - old_f1) / T)

	def _dt(self, t0, t1): return t1 - t0 if t0 is not None else 0

	def fit(self, X, y):
		if isinstance(X, pd.DataFrame):
			X = X.to_numpy()
		if isinstance(y, pd.DataFrame):
			y = y.to_numpy()
		elif isinstance(y, (list, pd.Series)):
			y = np.array(y)
		T = self.T
		T_min = self.T_min
		α = self.α
		max_iter = self.max_iter
		n_trans = self.n_trans
		grid = self.grid
		max_runtime = self.max_runtime
		cv = self.cv
		old_hyperparameters = {k: np.random.choice(v) if isinstance(v, list) else np.random.uniform(v[0], v[1]) for k, v in grid.items()}
		old_est = clone(self.estimator)
		old_est.set_params(**old_hyperparameters)
		old_f1, old_std = self._evaluate_f1(old_est, X, y, cv)
		best_f1 = old_f1
		best_hyperparameters = old_hyperparameters
		states_checked = {tuple(sorted(old_hyperparameters.items())): (old_f1, old_std)}
		total_iter = 1
		grid_f1s = [(1, T, old_f1, old_std, old_hyperparameters)]
		time_at_start = time.time()
		t_elapsed = self._dt(time_at_start, time.time())
		no_improvement_count = 0
		while T > T_min and total_iter < max_iter and t_elapsed < max_runtime and best_f1 < self.max_f1:
			for _ in range(self.n_trans):
				new_hyperparameters = self._generate_new_hyperparameters(old_hyperparameters, grid)
				new_f1, new_std = self._evaluate_f1_for_hyperparameters(new_hyperparameters, X, y, cv, states_checked)
				if new_f1 >= self.max_f1: break
				grid_f1s.append((total_iter, T, new_f1, new_std, new_hyperparameters))
				if new_f1 > best_f1:
					best_f1 = new_f1
					best_hyperparameters = new_hyperparameters
					no_improvement_count = 0
				else:
					no_improvement_count += 1
				print(f"{total_iter} T: {T:.5f}, f1: {new_f1:.6f}, std: {new_std:.6f}, hyperparameters: {new_hyperparameters}")
				if self._accept_prob(old_f1, new_f1, T) > np.random.random():
					old_hyperparameters = new_hyperparameters
					old_f1 = new_f1
				t_elapsed = self._dt(time_at_start, time.time())
				total_iter += 1
			if new_f1 >= self.max_f1:
				print(f"Max f1 reached {new_f1}!")
				break
			if no_improvement_count >= self.patience and T <= self.T_min:
				print("Early stopping due to lack of improvement.")
				break
			T *= α
		self.runtime_ = t_elapsed
		self.grid_f1s_ = grid_f1s
		self.best_f1_ = best_f1
		self.best_hyperparameters_ = best_hyperparameters

	def _generate_new_hyperparameters(self, old_hyperparameters, grid):
		new_hyperparameters = old_hyperparameters.copy()
		rand_key = np.random.choice(list(grid.keys()))
		val = grid[rand_key]
		if isinstance(val, list):
			sample_space = [v for v in val if v != old_hyperparameters[rand_key]]
			new_hyperparameters[rand_key] = np.random.choice(sample_space) if sample_space else np.random.choice(val)
		elif isinstance(val, tuple) and len(val) == 2:
			new_hyperparameters[rand_key] = np.random.uniform(val[0], val[1])
		return new_hyperparameters

	def _evaluate_f1_for_hyperparameters(self, hyperparameters, X, y, cv, states_checked):
		hyperparameters_tuple = tuple(sorted(hyperparameters.items()))
		if hyperparameters_tuple in states_checked:
			return states_checked[hyperparameters_tuple]
		else:
			est = clone(self.estimator)
			est.set_params(**hyperparameters)
			f1, std = self._evaluate_f1(est, X, y, cv)
			states_checked[hyperparameters_tuple] = (f1, std)
			return f1, std

	def _evaluate_f1(self, estimator, X, y, cv):
		if self.n_jobs > 1:
			out = Parallel(n_jobs=self.n_jobs)(
				delayed(_fit_and_score)(clone(estimator), X, y, self.scoring, train, test, verbose=True,
				                        parameters={}, fit_params={}, return_parameters=False, error_score="raise")
				for train, test in KFold(cv).split(X)
			)
		else:
			scores = []
			for train, test in KFold(cv).split(X):
				estimator.fit(X[train], y[train])
				y_pred = estimator.predict(X[test])
				scores.append(self.scoring(y[test], y_pred))
			out = (np.mean(scores), np.std(scores))
		return out

def cv_simulated_annealing(model_type, grid, T=10, T_min=0.001, α=0.9, n_trans=5, max_iter=100, max_runtime=60, cv=5, min_improvement=1e-4, patience=10):
	model_class, _ = MODEL_TYPE_TO_CLASS_TO_HYPERPARAMETER_MAP[model_type]
	model = model_class()
	sa = SA(
		estimator=model,
		grid=grid,
		scoring=f1_score,
		T=T,
		T_min=T_min,
		α=α,
		n_trans=n_trans,
		max_iter=max_iter,
		max_runtime=max_runtime,
		cv=cv,
		min_improvement=min_improvement,
		patience=patience
	)
	sa.fit(X_train, y_train)
	return sa.best_f1_, sa.best_hyperparameters_, sa.grid_f1s_, sa.runtime_