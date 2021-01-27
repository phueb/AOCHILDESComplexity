"""
Research questions:
1. Is partition 1 syntactically more complex? Does it have more unique POS-tag sequences (must be sentences)?
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from childescomplexity import configs
from childescomplexity.util import fit_line, split_into_sentences
from childescomplexity.util import split

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


y = []
for tags in split(prep.store.tokens, prep.num_tokens_in_part):

    sentences = split_into_sentences(tags, punctuation={'.'})
    unique_sentences = np.unique(sentences)
    print(f'Found {len(sentences):>12,} total sentences in part')
    print(f'Found {len(unique_sentences):>12,} unique sentences in part')

    y.append(len(unique_sentences))


# fig
_, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
plt.title('')
ax.set_ylabel('Num unique tag-sequences')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
x = np.arange(params.num_parts)
ax.plot(x, y, '-', alpha=0.5)
y_fitted = fit_line(x, y)
ax.plot(x, y_fitted, '-')
plt.show()

# fig
_, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
plt.title('Syntactic Complexity')
ax.set_ylabel(f'Z-scored Num unique tag-sequences')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.axhline(y=0, color='grey', linestyle=':')
x = np.arange(params.num_parts)
ax.plot(x, stats.zscore(y), alpha=1.0)
plt.show()