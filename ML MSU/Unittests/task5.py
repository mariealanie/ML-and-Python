import numpy as np

def get_part_of_array(X: np.ndarray) -> np.ndarray:
    return X[::4, 120:500:5]

def sum_non_neg_diag(X: np.ndarray) -> int:
    diag = np.diag(X)
    diag_non_neg = diag[diag >= 0]
    return int(diag_non_neg.sum()) if diag_non_neg.size > 0 else -1

def replace_values(X: np.ndarray) -> np.ndarray:
    X_copy = X.copy()  
    col_means = X_copy.mean(axis=0)  
    mask = (X_copy > 1.5 * col_means) | (X_copy < 0.25 * col_means) 
    X_copy[mask] = -1
    return X_copy