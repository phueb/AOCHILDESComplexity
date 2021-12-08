import numpy as np
from typing import Set, List
import random
from functools import reduce
from operator import iconcat
from scipy.stats import linregress

from aochildescomplexity import configs


def load_tokens(corpus_name: str,
                shuffle_docs: bool = False,
                shuffle_seed: int = 20,
                ) -> List[str]:

    """
    A "document" has type string. It is not tokenized.

    WARNING:
    Always use a seed for random operations.
    For example when loading tags and words using this function twice, they won't align if no seed is set

    WARNING:
    shuffling the documents does not remove all age-structure,
    because utterances associated with teh same age are still clustered within documents.
    """

    p = configs.Dirs.corpora / f'{corpus_name}.txt'
    text_in_file = p.read_text()

    docs = text_in_file.split('\n')

    num_docs = len(docs)
    print(f'Loaded {num_docs} documents from {corpus_name}')

    if shuffle_docs:
        random.seed(shuffle_seed)
        print('Shuffling documents')
        random.shuffle(docs)

    tokenized_docs = [d.split() for d in docs]
    res = reduce(iconcat, tokenized_docs, [])  # flatten list of lists

    return res


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


def partition(tokens: List[str],
              num_parts: int,
              ) -> List[List[str]]:
    res = []
    for start in np.linspace(0, len(tokens), num_parts)[:-1]:
        end = start + len(tokens) // num_parts
        token_part = tokens[int(start):int(end)]
        res.append(token_part)
    return res


def plot_best_fit_line(ax, x, y, color='red', zorder=3, x_pos=0.05, y_pos=0.9, plot_p=True):

    # fit line
    try:
        best_fit_fxn = np.polyfit(x, y, 1, full=True)
    except Exception as e:  # cannot fit line
        print('Cannot fit line.', e)
        return

    # make line
    slope = best_fit_fxn[0][0]
    intercept = best_fit_fxn[0][1]
    xl = [min(x), max(x)]
    yl = [slope * xx + intercept for xx in xl]

    # plot line
    ax.plot(xl, yl, linewidth=1, c=color, zorder=zorder)

    # plot rsqrd
    variance = np.var(y)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
    Rsqr = np.round(1 - residuals / variance, decimals=3)
    ax.text(x_pos,
            y_pos,
            '$R^2$ = {}'.format(Rsqr),
            transform=ax.transAxes,
            fontsize=configs.Fig.ax_fontsize-6)

    if plot_p:
        p = np.round(linregress(x, y)[3], decimals=8)
        ax.text(x_pos,
                y_pos - 0.05,
                'p = {:.4f}'.format(p),
                transform=ax.transAxes,
                fontsize=configs.Fig.ax_fontsize-6)
