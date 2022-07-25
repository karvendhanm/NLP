# manipulate n_gram count dictionary

import numpy as np
import pandas as pd
from collections import defaultdict

n_gram_counts = {
    ('i', 'am', 'happy'): 2,
    ('am', 'happy', 'because'): 1}

# get count of n-gram tuple.
print(f"count of n-gram ('i', 'am', 'happy'): {n_gram_counts[('i', 'am', 'happy')]}")

# Merge two tuples in python
prefix = ('i', 'am', 'happy')
word = 'because'

n_gram = prefix + (word,)


def single_pass_trigram_count_matrix(corpus, n_grams=3):
    """
    Creates the trigram count matrix from the input corpus in a single pass through the corpus.

    Args:
        corpus: Pre-processed and tokenized corpus.
        n_grams: number of words sequence (e.g: tri-gram, four-gram etc..,)

    Returns:
        bigrams: list of all bigram prefixes, row index
        vocabulary: list of all found words, the column index
        count_matrix: pandas dataframe with bigram prefixes as rows,
                      vocabulary words as columns
                      and the counts of the bigram/word combinations (i.e. trigrams) as values
    """
    bigrams = []
    vocabulary = []
    count_matrix_dict = defaultdict(dict)

    for i in range(len(corpus) - n_grams + 1):
        # the sliding window starts at position i and contains 3 words
        trigram = tuple(corpus[i: i + 3])

        bigram = trigram[0: -1]
        if not bigram in bigrams:
            bigrams.append(bigram)

        last_word = trigram[-1]
        if last_word not in vocabulary:
            vocabulary.append(last_word)

        if (bigram, last_word) not in count_matrix_dict:
            count_matrix_dict[(bigram, last_word)] = 0

        count_matrix_dict[(bigram, last_word)] += 1

    # convert the count matrix to np.array to fill in the blanks
    count_matrix = np.zeros((len(bigrams), len(vocabulary)))
    for trigram_key, trigram_count in count_matrix_dict.items():
        count_matrix[bigrams.index(trigram_key[0]), vocabulary.index(trigram_key[1])] = trigram_count

    # np.array to pandas dataframe conversion
    count_matrix = pd.DataFrame(count_matrix, index=bigrams, columns=vocabulary)
    return bigrams, vocabulary, count_matrix


corpus = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']
bigrams, vocabulary, count_matrix  = single_pass_trigram_count_matrix(corpus)
print(count_matrix)

# probability matrix

