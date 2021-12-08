"""
Research questions:
1. Is language in partition 1 more systematic or template-like?
"""

import numpy as np
from collections import Counter
from scipy import optimize
import matplotlib.pyplot as plt

from aochildescomplexity.utils import load_tokens
from aochildescomplexity.binned import make_age_bin2data, make_age_bin2data_with_min_size
from aochildescomplexity import configs
from aochildescomplexity.utils import plot_best_fit_line

CORPUS_NAME = 'childes-20201026'
SPLIT_SIZE = 5620
PLOT_FIT = False

# make equal-sized partitions corresponding to approximately equal sized age bins
age_bin2tokens_ = make_age_bin2data(CORPUS_NAME)
age_bin2tokens = make_age_bin2data_with_min_size(age_bin2tokens_)
num_bins = len(age_bin2tokens)

tokens = load_tokens(CORPUS_NAME)
type2id = {t: n for n, t in enumerate(set(tokens))}
num_types = len(type2id)


def fitfunc(p, x):
    return p[0] + p[1] * x


def errfunc(p, x, y):
    return y - fitfunc(p, x)


taylor_exponents = []
for part_id, (age_bin, tokens_part) in enumerate(age_bin2tokens.items()):

    token_ids = [type2id[t] for t in tokens_part]

    # make freq_mat
    num_tokens_in_part = len(token_ids)
    num_splits = num_tokens_in_part // SPLIT_SIZE + 1
    freq_mat = np.zeros((num_types, num_splits))
    start_locs = np.arange(0, num_tokens_in_part, SPLIT_SIZE)
    num_start_locs = len(start_locs)
    for split_id, start_loc in enumerate(start_locs):
        for token_id, f in Counter(token_ids[start_loc:start_loc + SPLIT_SIZE]).items():
            freq_mat[token_id, split_id] = f

    # x, y
    freq_mat = freq_mat[~np.all(freq_mat == 0, axis=1)]
    x = freq_mat.mean(axis=1)  # make sure not to have rows with zeros
    y = freq_mat.std(axis=1)

    # fit function + get alpha
    pinit = np.array([1.0, -1.0])
    logx = np.log10(x)
    logy = np.log10(y)
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=True)
    for i in out:
        print(i)
    pfinal = out[0]
    amp = pfinal[0]
    alpha = pfinal[1]

    # fig
    if PLOT_FIT:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=configs.Fig.dpi)
        plt.title(f'{CORPUS_NAME}\nnum_types={num_types:,}, part {part_id + 1} of {num_bins}')
        ax.set_xlabel('mean')
        ax.set_ylabel('std')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        # plot
        ax.text(x=1.0, y=0.3, s='Taylor\'s exponent: {:.3f}'.format(alpha))
        ax.loglog(x, y, '.', markersize=2)
        ax.loglog(x, amp * (x ** alpha) + 0, '.', markersize=2)
        plt.show()

    taylor_exponents.append(alpha)

# fig
_, ax = plt.subplots(figsize=(6, 4), dpi=configs.Fig.dpi)
ax.set_ylabel('Taylor Exponent', fontsize=configs.Fig.ax_fontsize)
ax.set_xlabel('Corpus Partition', fontsize=configs.Fig.ax_fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
x = np.arange(num_bins) + 1
ax.plot(x, taylor_exponents, '-')
plot_best_fit_line(ax, x, taylor_exponents, x_pos=0.70, y_pos=0.1)
plt.show()


for n, yi in enumerate(y):
    print(n + 1, yi)


