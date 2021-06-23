import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np
from childescomplexity.utils import load_tokens
from childescomplexity import configs
from childescomplexity.measures import calc_utterance_lengths

# /////////////////////////////////////////////////////////////////

CORPUS_NAME = 'childes-20201026'
WINDOW_SIZE = 10_000

tokens = load_tokens(CORPUS_NAME)

WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2
LW = 0.5

ys = [calc_utterance_lengths(tokens, rolling_avg=True, window_size=WINDOW_SIZE),
      calc_utterance_lengths(tokens, rolling_std=True, window_size=WINDOW_SIZE)]


# fig
y_labels = ['Mean\nUtt. Length', 'Std.\nUtt.Length']
fig, axs = plt.subplots(2, 1, dpi=configs.Fig.dpi, figsize=(6, 4))
for ax, y_label, y in zip(axs, y_labels, ys):
    if ax == axs[-1]:
        ax.set_xlabel('Corpus Location [# words]', fontsize=configs.Fig.ax_fontsize)
        x = np.arange(len(y)) + 1
        # ax.set_xticks(x)
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
    ax.plot(y, linewidth=LW, label=y_label, c='C0')
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()
