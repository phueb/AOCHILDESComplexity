import numpy as np
from typing import Optional, Set, List


def to_corr_mat(data_mat):
    mns = data_mat.mean(axis=1, keepdims=True)
    stds = data_mat.std(axis=1, ddof=1, keepdims=True) + 1e-6  # prevent np.inf (happens when dividing by zero)
    deviated = data_mat - mns
    zscored = deviated / stds
    res = np.matmul(zscored, zscored.T) / len(data_mat)  # it matters which matrix is transposed
    return res


def split_into_sentences(tokens: List[str],
                         punctuation: Set[str],
                         ) -> List[List[str]]:
    assert isinstance(punctuation, set)

    res = [[]]
    for w in tokens:
        res[-1].append(w)
        if w in punctuation:
            res.append([])
    return res