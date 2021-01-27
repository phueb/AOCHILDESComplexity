"""
Research questions:
1. How many unique windows are in the first vs second half of the input?
2. How many repeated windows are the first vs second half of the input?
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


WINDOW_SIZES = [2, 3, 4, 5, 6]

tokens1 = prep.store.tokens[:prep.midpoint]
tokens2 = prep.store.tokens[-prep.midpoint:]

 # todo this fn is in preppy
windows_mat1 = make_windows_mat(tokens1, prep.num_windows_in_part, prep.num_tokens_in_window)
windows_mat2 = make_windows_mat(tokens2, prep.num_windows_in_part, prep.num_tokens_in_window)


def calc_y(w_mat, w_size, uniq):
    truncated_w_mat = w_mat[:, -w_size:]
    u = np.unique(truncated_w_mat, axis=0)
    num_total_windows = len(truncated_w_mat)
    num_uniq = len(u)
    num_repeated = num_total_windows - num_uniq
    #
    print(num_total_windows, num_uniq, num_repeated)
    if uniq:
        return num_uniq
    else:
        return num_repeated


def plot(y_label, ys_list):
    bar_width0 = 0.0
    bar_width1 = 0.25
    _, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Window size')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    num_conditions = len(WINDOW_SIZES)
    xs = np.arange(1, num_conditions + 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(WINDOW_SIZES)
    # plot
    colors = sns.color_palette("hls", 2)[::-1]
    labels = ['partition 1', 'partition 2']
    for n, (x, ys) in enumerate(zip(xs, ys_list)):
        ax.bar(x + bar_width0, ys[0], bar_width1, color=colors[0], label=labels[0] if n == 0 else '_nolegend_')
        ax.bar(x + bar_width1, ys[1], bar_width1, color=colors[1], label=labels[1] if n == 0 else '_nolegend_')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


plot('Number of repeated windows', [(calc_y(windows_mat1, ws, False), calc_y(windows_mat2, ws, False))
                                    for ws in WINDOW_SIZES])

plot('Number of unique windows', [(calc_y(windows_mat1, ws, True), calc_y(windows_mat2, ws, True))
                                  for ws in WINDOW_SIZES])