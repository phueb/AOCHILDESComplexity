"""
Research questions:
1. does partition 1 have fewer unique noun chunks?
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import spacy
import pyprind

from childescomplexity import configs
from childescomplexity.util import fit_line, split_into_sentences
from childescomplexity.util import split
from childescomplexity.svo import subject_verb_object_triples

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


VERBOSE = True

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=['ner'])


y = []
pbar = pyprind.ProgBar(NUM_PARTS, stream=2) if not VERBOSE else None
for tokens in split(prep.store.tokens, prep.num_tokens_in_part):
    sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in sentences]

    triples_in_part = []
    for doc in nlp.pipe(texts):
        triples = [t for t in subject_verb_object_triples(doc)]  # only returns triples, not partial triples
        triples_in_part += triples
    num_triples_in_part = len(triples_in_part)
    num_unique_triples_in_part = len(set(triples_in_part))

    print(triples_in_part[:20])
    yi = num_unique_triples_in_part / num_triples_in_part

    if VERBOSE:
        print(f'Found {num_triples_in_part:>12,} SVO triples')
        print(f'Found {num_unique_triples_in_part:>12,} unique SVO triples')
        print(f'sem-complexity={yi:.4f}')
    else:
        pbar.update()

    y.append(yi)


# fig
_, ax = plt.subplots(figsize=configs.Fig.fig_size, dpi=configs.Fig.dpi)
plt.title('SVO-triples')
ax.set_ylabel('Num unique SVO-triples')
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
plt.title('SVO-triples')
ax.set_ylabel(f'Z-scored Num unique SVO-triples')
ax.set_xlabel('Partition')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
ax.axhline(y=0, color='grey', linestyle=':')
x = np.arange(params.num_parts)
ax.plot(x, stats.zscore(y), alpha=1.0)
plt.show()