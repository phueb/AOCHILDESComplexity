import numpy as np
from itertools import groupby
from typing import List, Dict

from childescomplexity import configs


def make_age_bin2data(corpus_name: str,
                      age_step: int = configs.Binning.age_step,
                      verbose: bool = False,
                      ) -> Dict[float, List[str]]:

    ages_path = configs.Dirs.corpora / f'{corpus_name}_ages.txt'
    ages_text = ages_path.read_text(encoding='utf-8')
    ages = np.array(ages_text.split(), dtype=np.float)
    data_path = configs.Dirs.corpora / f'{corpus_name}.txt'
    data_text = data_path.read_text(encoding='utf-8')
    data_by_doc = [doc.split() for doc in data_text.split('\n')[:-1]]
    ages_binned = ages - np.mod(ages, age_step)

    # convert ages to age bins
    ages_binned = ages_binned.astype(np.int)
    age_and_docs = zip(ages_binned, data_by_doc)

    res = {}
    for age_bin, doc_group in groupby(age_and_docs, lambda d: d[0]):
        docs = [d[1] for d in doc_group]
        data = list(np.concatenate(docs))
        if verbose:
            print(f'Found {len(docs)} transcripts for age-bin={age_bin}')

        res[age_bin] = data

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
