"""
Research questions:
1. Is partition 1 syntactically more complex? Does it have more unique POS-tag sequences (must be sentences)?


The number of unique POS-tag sequences should be larger in earlier partitions,
because sentences are shorter and therefore more numerous.
However, because syntactic complexity is much reduced in early partitions, this bias is overcome.

"""


import numpy as np
import matplotlib.pyplot as plt
import spacy
import pyprind

from aochildescomplexity.binned import make_age_bin2data, make_age_bin2data_with_min_size
from aochildescomplexity import configs
from aochildescomplexity.utils import plot_best_fit_line, split_into_sentences

CORPUS_NAME = 'childes-20201026'

# make equal-sized partitions corresponding to approximately equal sized age bins
age_bin2tokens_ = make_age_bin2data(CORPUS_NAME)
age_bin2tokens = make_age_bin2data_with_min_size(age_bin2tokens_)
num_bins = len(age_bin2tokens)

nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

y = []
pbar = pyprind.ProgBar(num_bins, stream=2)
for _, tags in age_bin2tokens.items():

    sentences = split_into_sentences(tags, punctuation={'.', '?', '!'})
    sentences_as_strings = [' '.join(s) for s in sentences]

    # get POS-tag sequences
    unique_pos_tag_sequences = set()
    for doc in nlp.pipe(sentences_as_strings):
        pos_tag_sequence = ' '.join([t.pos_ for t in doc])
        unique_pos_tag_sequences.add(pos_tag_sequence)

    y.append(len(unique_pos_tag_sequences))

    pbar.update()


# fig
_, ax = plt.subplots(figsize=(6, 4), dpi=configs.Fig.dpi)
plt.title('')
ax.set_ylabel('Number of POS-tag\nsequence types', fontsize=configs.Fig.ax_fontsize)
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
    print(n + 1, yi)