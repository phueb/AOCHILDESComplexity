import numpy as np
from itertools import groupby
from typing import List, Dict

from aochildes.pipeline import Pipeline

from aochildescomplexity import configs


def make_age_bin2data(age_step: int = configs.Binning.age_step,
                      verbose: bool = False,
                      ) -> Dict[float, List[str]]:

    # load transcripts from AO-CHILDES corpus
    pipeline = Pipeline()
    transcripts = pipeline.load_age_ordered_transcripts()

    # convert ages to age bins
    ages = np.array([t.age for t in transcripts])
    ages_binned = ages - np.mod(ages, age_step)
    ages_binned = ages_binned.astype(np.int)

    res = {}
    ages_and_transcripts = zip(ages_binned, transcripts)
    for age_bin, transcript_group in groupby(ages_and_transcripts, lambda d: d[0]):
        ts = [d[1] for d in transcript_group]
        if verbose:
            print(f'Found {len(ts)} transcripts for age-bin={age_bin}')

        res[age_bin] = [w for t in ts for w in t.text.split()]  # list of all words in transcript

    return res


def make_age_bin2data_with_min_size(age_bin2data: Dict[float, List[str]],
                                    min_num_data: int = configs.Binning.num_tokens_in_bin,
                                    verbose: bool = False,
                                    ):
    """
    return dictionary similar to age_bin2tokens but with a constant number of data per age_bin.
    combine bins when a bin is too small.
    """

    res = {}
    buffer = []
    for age_bin, data in age_bin2data.items():

        buffer += data

        if len(buffer) > min_num_data:
            res[age_bin] = buffer[-min_num_data:]
            buffer = []
        else:
            continue

    # printout
    if verbose:
        for age_bin, tokens in res.items():
            print(f'age start (days)={age_bin:>12} num tokens in bin={len(tokens):,}')

    return res
