"""
Research questions:
1. Is language in partition 1 more systematic or template-like?
"""

import numpy as np
from collections import Counter
from scipy import optimize
import matplotlib.pyplot as plt

from childescomplexity import configs

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20201026'
AGE_STEP = 100
NUM_TOKENS_PER_BIN = 100_000  # 100K is good with AGE_STEP=100


age_bin2tokens_ = make_age_bin2data(CORPUS_NAME, AGE_STEP, suffix='')
for word_tokens in age_bin2tokens_.values():  # this is used to determine maximal NUM_TOKENS_PER_BIN
    print(f'{len(word_tokens):,}')
# combine small bins
age_bin2tokens = make_age_bin2data_with_min_size(age_bin2tokens_, NUM_TOKENS_PER_BIN)
num_bins = len(age_bin2tokens)

# /////////////////////////////////////////////////////////////////



SPLIT_SIZE = 5620
PLOT_FIT = False


def fitfunc(p, x):
    return p[0] + p[1] * x


def errfunc(p, x, y):
    return y - fitfunc(p, x)




for part_id, part in enumerate(prep.reordered_parts):
    # make freq_mat
    num_splits = prep.num_tokens_in_part // SPLIT_SIZE + 1
    freq_mat = np.zeros((prep.store.num_types, num_splits))
    start_locs = np.arange(0, prep.num_tokens_in_part, SPLIT_SIZE)
    num_start_locs = len(start_locs)
    for split_id, start_loc in enumerate(start_locs):
        for token_id, f in Counter(part[start_loc:start_loc + SPLIT_SIZE]).items():
            freq_mat[token_id, split_id] = f
    # x, y
    freq_mat = freq_mat[~np.all(freq_mat == 0, axis=1)]
    x = freq_mat.mean(axis=1)  # make sure not to have rows with zeros
    y = freq_mat.std(axis=1)
    # fit
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
    fig, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
    plt.title(f'{CORPUS_NAME}\nnum_types={NUM_TYPES:,}, part {part_id + 1} of {NUM_PARTS}')
    ax.set_xlabel('mean')
    ax.set_ylabel('std')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # plot
    ax.text(x=1.0, y=0.3, s='Taylor\'s exponent: {:.3f}'.format(alpha))
    ax.loglog(x, y, '.', markersize=2)
    if PLOT_FIT:
        ax.loglog(x, amp * (x ** alpha) + 0, '.', markersize=2)  # TODO test
    plt.show()






