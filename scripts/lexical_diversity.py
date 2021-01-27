import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


from childescomplexity import configs
from childescomplexity.measures import calc_entropy
from childescomplexity.measures import mtld

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


AX_FONTSIZE = 8
LEG_FONTSIZE = 6
FIGSIZE = (3.2, 2.2)
DPI = configs.Fig.dpi
IS_LOG = True
WSPACE = 0.0
HSPACE = 0.0
WPAD = 0.0
HPAD = 0.0
PAD = 0.2
LW = 0.5

# xys
ys = [
    [calc_entropy(part) for part in prep.reordered_parts],
    [mtld(part) for part in prep.reordered_parts]
]

# fig
y_labels = ['Shannon Entropy', 'MTLD']
fig, axs = plt.subplots(2, 1, dpi=configs.Fig.dpi, figsize=configs.Fig.fig_size)
for ax, y_label, y in zip(axs, y_labels, ys):
    if ax == axs[-1]:
        ax.set_xlabel('Corpus Location', fontsize=AX_FONTSIZE, labelpad=-10)
        ax.set_xticks([0, len(y)])
        ax.set_xticklabels(['0', f'{prep.store.num_tokens:,}'])
        plt.setp(ax.get_xticklabels(), fontsize=AX_FONTSIZE)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(y_label, fontsize=LEG_FONTSIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    plt.setp(ax.get_yticklabels(), fontsize=LEG_FONTSIZE)
    # plot
    ax.plot(y, linewidth=LW, label=y_label, c='black')
# show
plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
plt.tight_layout(h_pad=HPAD, w_pad=WPAD, pad=PAD)
plt.show()
