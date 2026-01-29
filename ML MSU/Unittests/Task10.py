import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):
    def __init__(self, dtype=np.float64):
        super().__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        self.categories_ = [np.sort(X[col].unique()) for col in X.columns]

    def transform(self, X):
        encoded_columns = []
        for i, col in enumerate(X.columns):
            unique_vals = self.categories_[i]
            for val in unique_vals:
                encoded_columns.append((X[col] == val).astype(self.dtype))
        return np.column_stack(encoded_columns)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        self.success_rates = []
        self.frequencies = []
        for col in X.columns:
            success_dict = {}
            freq_dict = {}
            unique_vals = X[col].unique()

            def process_value(val):
                mask = (X[col] == val)
                success_val = Y[mask].mean()
                freq_val = mask.mean()
                return success_val, freq_val

            vectorized_processor = np.vectorize(process_value)
            results = vectorized_processor(unique_vals)

            for i, val in enumerate(unique_vals):
                success_dict[val] = results[0][i]
                freq_dict[val] = results[1][i]

            self.success_rates.append(success_dict)
            self.frequencies.append(freq_dict)

    def transform(self, X, a=1e-5, b=1e-5):
        transformed = []
        for i, col in enumerate(X.columns):
            col_data = X[col].values

            def get_success(val):
                return self.success_rates[i].get(val, 0.0)

            def get_freq(val):
                return self.frequencies[i].get(val, 0.0)

            v_success = np.vectorize(get_success)
            v_freq = np.vectorize(get_freq)
            success_vals = v_success(col_data)
            freq_vals = v_freq(col_data)

            def calculate_ratio(s, f):
                return (s + a) / (f + b)

            v_ratio = np.vectorize(calculate_ratio)
            ratio_vals = v_ratio(success_vals, freq_vals)
            transformed.extend([success_vals, freq_vals, ratio_vals])
        return np.column_stack(transformed).astype(self.dtype)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[: i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[: (n_splits - 1) * n_]


class FoldCounters:
    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        self.folds_info = list(group_k_fold(len(X), self.n_folds, seed))
        self.fold_stats = []

        for fold_idx, (test_idx, train_idx) in enumerate(self.folds_info):
            X_train = X.iloc[train_idx]
            Y_train = Y.iloc[train_idx]
            fold_metrics = []

            for col in X.columns:
                success_dict = {}
                freq_dict = {}

                unique_vals = X_train[col].unique()

                def process_value(val):
                    mask = (X_train[col] == val)
                    if mask.sum() > 0:
                        success_val = Y_train[mask].mean()
                        freq_val = mask.mean()
                    else:
                        success_val = 0.0
                        freq_val = 0.0
                    return success_val, freq_val

                vectorized_processor = np.vectorize(process_value)
                results = vectorized_processor(unique_vals)

                for i, val in enumerate(unique_vals):
                    success_dict[val] = results[0][i]
                    freq_dict[val] = results[1][i]
                fold_metrics.append((success_dict, freq_dict))

            self.fold_stats.append(fold_metrics)

    def transform(self, X, a=1e-5, b=1e-5):
        result = []
        for obj_idx in range(len(X)):
            obj_features = []
            for fold_idx, (test_idx, train_idx) in enumerate(self.folds_info):

                if obj_idx in test_idx:
                    for feat_idx, col in enumerate(X.columns):
                        val = X.iloc[obj_idx][col]
                        success_dict, freq_dict = self.fold_stats[fold_idx][feat_idx]
                        success = success_dict.get(val, 0.0)
                        freq = freq_dict.get(val, 0.0)

                        def calculate_ratio(s, f):
                            return (s + a) / (f + b) if (f + b) != 0 else 0.0

                        v_ratio = np.vectorize(calculate_ratio)
                        ratio = v_ratio(np.array([success]), np.array([freq]))[0]
                        obj_features.extend([success, freq, ratio])
                    break

            result.append(obj_features)
        return np.array(result, dtype=self.dtype)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    unique_vals = np.unique(x)

    def process_value(val):
        mask = (x == val)
        y_subset = y[mask]
        if len(y_subset) == 0 or np.all(y_subset == 0):
            return 0.0
        elif np.all(y_subset == 1):
            return 1.0
        else:
            return y_subset.mean()

    vectorized_processor = np.vectorize(process_value)
    return vectorized_processor(unique_vals)
