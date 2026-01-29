from typing import List
from copy import deepcopy

def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    result = []
    for i in range(0, len(X), 4):
        row = []
        for j in range(120, 500, 5):
            row.append(X[i][j])
        result.append(row)
    return result

def sum_non_neg_diag(X: List[List[int]]) -> int:
    diag_sum = 0
    found = False
    n = len(X)
    m = len(X[0])
    for i in range(min(n, m)):
        if X[i][i] >= 0:
            diag_sum += X[i][i]
            found = True
    return diag_sum if found else -1

def replace_values(X: List[List[float]]) -> List[List[float]]:
    X_copy = deepcopy(X)
    n = len(X_copy)
    m = len(X_copy[0])
    col_means = []
    for j in range(m):
        s = 0
        for i in range(n):
            s += X_copy[i][j]
        col_means.append(s / n)
    for j in range(m):
        for i in range(n):
            if X_copy[i][j] > 1.5 * col_means[j] or X_copy[i][j] < 0.25 * col_means[j]:
                X_copy[i][j] = -1
    return X_copy
