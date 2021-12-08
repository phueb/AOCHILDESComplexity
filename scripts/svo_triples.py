"""
Research questions:
1. does partition 1 have fewer unique noun chunks?
"""


import numpy as np
import matplotlib.pyplot as plt
import spacy
import pyprind

from aochildescomplexity.binned import make_age_bin2data, make_age_bin2data_with_min_size
from aochildescomplexity import configs
from aochildescomplexity.utils import plot_best_fit_line, split_into_sentences
from aochildescomplexity.svo import subject_verb_object_triples

CORPUS_NAME = 'childes-20201026'
VERBOSE = False

# make equal-sized partitions corresponding to approximately equal sized age bins
age_bin2tokens_ = make_age_bin2data(CORPUS_NAME)
age_bin2tokens = make_age_bin2data_with_min_size(age_bin2tokens_)
num_bins = len(age_bin2tokens)

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=['ner'])


y = []
pbar = pyprind.ProgBar(num_bins, stream=2) if not VERBOSE else None
for _, tokens in age_bin2tokens.items():
    sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in sentences]

    triples_in_part = []
    for doc in nlp.pipe(texts):
        triples = [t for t in subject_verb_object_triples(doc)]  # only returns triples, not partial triples
        triples_in_part += triples
    num_triples_in_part = len(triples_in_part)
    num_unique_triples_in_part = len(set(triples_in_part))

    yi = num_unique_triples_in_part / num_triples_in_part

    if VERBOSE:
        print(triples_in_part[:20])
        print(f'Found {num_triples_in_part:>12,} SVO triples')
        print(f'Found {num_unique_triples_in_part:>12,} unique SVO triples')
        print(f'sem-complexity={yi:.4f}')
    else:
        pbar.update()

    y.append(yi)


# fig
_, ax = plt.subplots(figsize=(6, 4), dpi=configs.Fig.dpi)
ax.set_ylabel('SVO-triple types / tokens', fontsize=configs.Fig.ax_fontsize)
ax.set_xlabel('Partition', fontsize=configs.Fig.ax_fontsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
x = np.arange(num_bins) + 1
ax.plot(x, y, '-')
plot_best_fit_line(ax, x, y, x_pos=0.70, y_pos=0.1)
plt.show()

for n, yi in enumerate(y):
    print(n, yi)