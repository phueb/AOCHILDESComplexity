import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from childescomplexity.binned import make_age_bin2data, make_age_bin2data_with_min_size
from childescomplexity import configs
from childescomplexity.measures import calc_entropy
from childescomplexity.measures import mtld


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

ys = [
    [calc_entropy(part) for part in age_bin2tokens.values()],
    [mtld(part) for part in age_bin2tokens.values()]
]

# fig
y_labels = ['Shannon\nEntropy', 'MTLD']
fig, axs = plt.subplots(2, 1, dpi=configs.Fig.dpi, figsize=(6, 4))
x = np.arange(num_bins) + 1
for ax, y_label, y in zip(axs, y_labels, ys):
    if ax == axs[-1]:
        ax.set_xlabel('Corpus Partition', fontsize=configs.Fig.ax_fontsize)
        ax.set_xticks(x)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(y_label, fontsize=configs.Fig.ax_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    plt.setp(ax.get_yticklabels(), fontsize=configs.Fig.leg_fontsize)
    # plot
    ax.plot(x, y, linewidth=1, label=y_label, c='C0')
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()
