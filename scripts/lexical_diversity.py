import matplotlib.pyplot as plt
import numpy as np

from aochildescomplexity.binned import make_age_bin2data, make_age_bin2data_with_min_size
from aochildescomplexity import configs
from aochildescomplexity.utils import plot_best_fit_line
from aochildescomplexity.measures import mtld


CORPUS_NAME = 'childes-20201026'
NUM_PARTS = 16

# make equal-sized partitions corresponding to approximately equal sized age bins
age_bin2tokens_ = make_age_bin2data(CORPUS_NAME)
age_bin2tokens = make_age_bin2data_with_min_size(age_bin2tokens_)
num_bins = len(age_bin2tokens)

WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2

y = []
for part in age_bin2tokens.values():
    y.append(mtld(part))

# fig
_, ax = plt.subplots(figsize=(6, 4), dpi=configs.Fig.dpi)
ax.set_ylabel('Measure of\nLexical Textual Diversity', fontsize=configs.Fig.ax_fontsize)
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
    print(n + 1, yi)