import numpy as np
from collections import Counter


def calc_entropy(words):  # this does not show a strong age-related trend compared to MTLD
    num_labels = len(words)
    probabilities = np.asarray([count / num_labels for count in Counter(words).values()])
    result = - probabilities.dot(np.log2(probabilities))
    return result


def safe_divide(numerator, denominator):
    if denominator == 0 or denominator == 0.0:
        index = 0
    else:
        index = numerator/denominator
    return index


def ttr(text):
    ntokens = len(text)
    ntypes = len(set(text))

    return safe_divide(ntypes, ntokens)


def mtld(words, min=10):
    """
    Measure of lexical textual diversity (MTLD), described in Jarvis & McCarthy.
    'average number of words until text is saturated, or stable'
    implementation obtained from
    https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
    """
    def mtlder(text):
        factor = 0
        factor_lengths = 0
        start = 0
        for x in range(len(text)):
            factor_text = text[start:x+1]
            if x+1 == len(text):
                factor += safe_divide((1 - ttr(factor_text)),(1 - .72))
                factor_lengths += len(factor_text)
            else:
                if ttr(factor_text) < .720 and len(factor_text) >= min:
                    factor += 1
                    factor_lengths += len(factor_text)
                    start = x+1
                else:
                    continue

        mtld = safe_divide(factor_lengths,factor)
        return mtld
    input_reversed = list(reversed(words))
    res = safe_divide((mtlder(words) + mtlder(input_reversed)), 2)
    return res


