"""
Research questions:
1. does partition 1 have fewer noun chunk types? in other words, are fewer meanings expressed?
"""


import numpy as np
import matplotlib.pyplot as plt
import spacy
import pyprind

from aochildescomplexity.binned import make_age_bin2data, make_age_bin2data_with_min_size
from aochildescomplexity import configs
from aochildescomplexity.utils import plot_best_fit_line, split_into_sentences

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
for age_bin, tokens in age_bin2tokens.items():
    sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
    texts = [' '.join(s) for s in sentences]

    noun_chunks_in_part = []
    for doc in nlp.pipe(texts):
        for chunk in doc.noun_chunks:
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
_, ax = plt.subplots(figsize=(6, 4), dpi=configs.Fig.dpi)
ax.set_ylabel('Number of\nnoun chunk types', fontsize=configs.Fig.ax_fontsize)
ax.set_xlabel('Corpus Partition', fontsize=configs.Fig.ax_fontsize)
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