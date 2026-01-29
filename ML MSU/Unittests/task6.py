from collections import Counter
from typing import List

def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    return Counter(x) == Counter(y)

def max_prod_mod_3(x: List[int]) -> int:
    max_prod = -1
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if a % 3 == 0 or b % 3 == 0:
            prod = a * b
            if prod > max_prod:
                max_prod = prod
    return max_prod

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    h = len(image)
    w = len(image[0])
    c = len(weights)
    result = [[0.0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            s = 0.0
            for k in range(c):
                s += image[i][j][k] * weights[k]
            result[i][j] = s
    return result

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    vec_x = []
    for val, count in x:
        vec_x.extend([val] * count)
    vec_y = []
    for val, count in y:
        vec_y.extend([val] * count)

    if len(vec_x) != len(vec_y):
        return -1

    return sum(a * b for a, b in zip(vec_x, vec_y))

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    def cos_between_vectors(v1: List[float], v2: List[float]) -> float:
        lenth = len(v1)
        scalar = 0
        for i in range(lenth):
            scalar += v1[i] * v2[i]
        
        norm_v1 = 0
        for i in range(lenth):
            norm_v1 += v1[i] ** 2
        norm_v1 = norm_v1 ** 0.5

        norm_v2 = 0
        for i in range(lenth):
            norm_v2 += v2[i] ** 2
        norm_v2 = norm_v2 ** 0.5
        if norm_v1 == 0 or norm_v2 == 0:
            return 1
        return scalar / (norm_v1 * norm_v2)
    
    ans = []
    for i in range(len(X)):
        arr = []
        for j in range(len(Y)):
            arr.append(cos_between_vectors(X[i], Y[j]))
        ans.append(arr)
    return ans