import numpy as np
import pandas as pd
from collections import defaultdict


def single_pass_trigram_count_matrix(corpus):
    """
        Creates the trigram count matrix from the input corpus in a single pass through the corpus.

        Args:
            corpus: Pre-processed and tokenized corpus.

        Returns:
            bigrams: list of all bigram prefixes, row index
            vocabulary: list of all found words, the column index
            count_matrix: pandas dataframe with bigram prefixes as rows,
                          vocabulary words as columns
                          and the counts of the bigram/word combinations (i.e. trigrams) as values
    """

    bigrams = []
    vocabulary = []
    count_matrix_dict = defaultdict(int)

    # go through the corpus once with a sliding window
    what_gram = 3  # the n gram that we are trying to construct
    for i in range(len(corpus) - what_gram + 1):
        trigram = tuple(corpus[i: i + what_gram])
        bigram = trigram[0: -1]
        if bigram not in bigrams:
            bigrams.append(bigram)

        last_word = trigram[-1]
        if not last_word in vocabulary:
            vocabulary.append(last_word)
        count_matrix_dict[bigram, last_word] += 1

    # convert the count matrix to np.array to fill in the blanks
    count_matrix = np.zeros((len(bigrams), len(vocabulary)))
    for trigram_key, trigram_count in count_matrix_dict.items():
        count_matrix[bigrams.index(trigram_key[0]), vocabulary.index(trigram_key[1])] = trigram_count

    # np.array to pandas dataframe conversion
    count_matrix = pd.DataFrame(count_matrix, index=bigrams, columns=vocabulary)
    return bigrams, vocabulary, count_matrix






