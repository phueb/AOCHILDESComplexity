"""
Research questions:
1. does partition 1 have fewer noun chunks? in other words, are fewer meanings expressed?
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import spacy
from spacy.tokens import Span
import pyprind

from childescomplexity.binned import make_age_bin2data, make_age_bin2data_with_min_size
from childescomplexity import configs
from childescomplexity.util import fit_line, split_into_sentences

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


VERBOSE = False

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=['ner'])


def contains_symbol(span):
    """
    checks if span has any undesired symbols.
    used to filter noun chunks.
    """
    return any(s in span.text for s in set(configs.Symbols.all))


Span.set_extension("contains_symbol", getter=contains_symbol)


y = []
pbar = pyprind.ProgBar(num_bins, stream=2) if not VERBOSE else None
for age_bin, tokens in age_bin2tokens.items():
    sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in sentences]

    noun_chunks_in_part = []
    for doc in nlp.pipe(texts):
        for chunk in doc.noun_chunks:
            if not chunk._.contains_symbol:
                noun_chunks_in_part.append(chunk.text)

    num_chunks_in_part = len(noun_chunks_in_part)
    num_unique_chunks_in_part = len(set(noun_chunks_in_part))
    if VERBOSE:
        print(f'Found {num_chunks_in_part:>12,} noun chunks')
        print(f'Found {num_unique_chunks_in_part:>12,} unique noun chunks')
    else:
        pbar.update()

    y.append(num_unique_chunks_in_part)


# fig
_, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
plt.title('Noun chunks')
ax.set_ylabel('Num unique noun chunks')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
x = np.arange(num_bins)
ax.plot(x, y, '-', alpha=0.5)
y_fitted = fit_line(x, y)
ax.plot(x, y_fitted, '-')
plt.show()

# fig
_, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
plt.title('Noun chunks')
ax.set_ylabel(f'Z-scored Num unique noun chunks')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.axhline(y=0, color='grey', linestyle=':')
x = np.arange(num_bins)
ax.plot(x, stats.zscore(y), alpha=1.0)
plt.show()