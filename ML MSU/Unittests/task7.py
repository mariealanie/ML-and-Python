import numpy as np

def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size != y.size:
        return False
    return np.array_equal(np.sort(x), np.sort(y))

def max_prod_mod_3(x: np.ndarray) -> int:
    products = x[:-1] * x[1:]
    mask = (x[:-1] % 3 == 0) | (x[1:] % 3 == 0)
    valid_products = products[mask]
    if len(valid_products) == 0:
        return -1
    return np.max(valid_products)

def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(image * weights, axis=2)    

def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x_values = np.repeat(x[:, 0], x[:, 1])
    y_values = np.repeat(y[:, 0], y[:, 1])
    if x_values.size != y_values.size:
        return -1
    return np.dot(x_values, y_values)

def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_norms = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norms = np.linalg.norm(Y, axis=1, keepdims=True)
    zero_mask = (X_norms == 0) | ((Y_norms == 0).T)
    X_norms[X_norms == 0] = 1
    Y_norms[Y_norms == 0] = 1
    X_normalized = X / X_norms
    Y_normalized = Y / Y_norms
    cos_dist = np.dot(X_normalized, Y_normalized.T)
    cos_dist[zero_mask] = 1
    return cos_dist