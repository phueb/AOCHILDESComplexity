from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    scripts = root / 'scripts'
    corpora = root / 'corpora'


class Fig:
    ax_fontsize = 18
    leg_fontsize = 12
    dpi = 200


class Binning:
    age_step = 100
    num_tokens_in_bin = 100_000  # 100K is good with AGE_STEP=100
