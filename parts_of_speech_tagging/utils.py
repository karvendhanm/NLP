from collections import defaultdict
import math
import numpy as np
import operator
import string


def assign_unk(word):
    """
    Assign tokens to unknown words
    """

    # Punctuation characters
    # Try printing them out in a new cell!
    punct = set(string.punctuation)

    # Suffixes
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling",
                   "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]

    # Loop the characters in the word, check if any is a digit
    if any(char.isdigit() for char in word):
        return "--unk_digit--"

    # Loop the characters in the word, check if any is a punctuation character
    elif any(char in punct for char in word):
        return "--unk_punct--"

    # Loop the characters in the word, check if any is an upper case character
    elif any(char.isupper() for char in word):
        return "--unk_upper--"

    # Check if word ends with any noun suffix
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Check if word ends with any verb suffix
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Check if word ends with any adjective suffix
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Check if word ends with any adverb suffix
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    # If none of the previous criteria is met, return plain unknown
    return "--unk--"


def get_word_tag(line, vocab):
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        word, tag = line.split()
        if word not in vocab:
            word = assign_unk(word)
    return word, tag


def preprocess(vocab, data_fp):
    '''
    preprocess data
    :param vocab:
    :param data_fp:
    :return:
    '''

    orig = []
    prep = []

    # Read data
    with open(data_fp, 'r') as data_file:

        for cnt, word in enumerate(data_file):

            orig.append(word.strip())
            # End of sentence
            if not word.split():
                prep.append("--n--")
                continue

            # Handle unknown words
            elif word.strip() not in vocab:
                word = assign_unk(word.strip())
                prep.append(word)
                continue

            else:
                prep.append(word.strip())

    assert len(orig) == len(open(data_fp, 'r').readlines())
    assert len(prep) == len(open(data_fp, 'r').readlines())

    return orig, prep


def create_dictionaries(training_corpus, vocab, verbose=True):
    """
    Input:
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """

    # initialize the dictionaries using default dict:
    transition_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    prev_tag = "--s--"

    for _idx, word_tag in enumerate(training_corpus):

        if _idx % 50000 == 0 and verbose:
            print(f'processing word_count = {_idx}')

        word, tag = get_word_tag(word_tag, vocab)

        transition_counts[(prev_tag, tag)] += 1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1

        prev_tag = tag

    return emission_counts, transition_counts, tag_counts


# My implementation
def predict_pos1(prep, y, emission_counts, vocab, states):
    '''
        Input:
            prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
            y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
            emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
            vocab: a dictionary where keys are words in vocabulary and value is an index
            states: a sorted list of all possible tags for this assignment
        Output:
            accuracy: Number of times you classified a word correctly
    '''

    _dict = defaultdict(list)
    for (tag, word), count in emission_counts.items():

        if word in prep:
            _dict[word].append((tag, count))

    _word_high_freq_tag = {}
    for key, val in _dict.items():
        most_used_tag = sorted(val, key=operator.itemgetter(1), reverse=True)[0][0]
        _word_high_freq_tag[key] = most_used_tag

    cnt = 0
    for line in y:
        if not line.split():
            cnt += 1
            continue

        train_word, train_tag = line.split()
        if train_tag == _word_high_freq_tag.get(train_word, "UNKNOWN"):
            cnt += 1

    return cnt / len(prep)


# Coursera implementation
def predict_pos2(prep, y, emission_counts, vocab, states):
    '''
    Input:
        prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
        y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags for this assignment
    Output:
        accuracy: Number of times you classified a word correctly
    '''

    # Initialize the number of correct predictions to zero
    num_correct = 0

    # Get the (tag, word) tuples, stored as a set
    all_words = set(emission_counts.keys())

    # Get the number of (word, POS) tuples in the corpus 'y'
    total = len(y)
    for word, y_tup in zip(prep, y):

        # Split the (word, POS) string into a list of two items
        y_tup_l = y_tup.split()

        # Verify that y_tup contain both word and POS
        if len(y_tup_l) == 2:

            # Set the true POS label for this word
            true_label = y_tup_l[1]

        else:
            # If the y_tup didn't contain word and POS, go to next word
            continue

        count_final = 0
        pos_final = ''

        # If the word is in the vocabulary...
        if word in vocab:
            for pos in states:

                # define the key as the tuple containing the POS and word
                key = (pos, word)

                # check if the (pos, word) key exists in the emission_counts dictionary
                if key in emission_counts:  # Replace None in this line with the proper condition.

                    # get the emission count of the (pos,word) tuple
                    count = emission_counts[key]

                    # keep track of the POS with the largest count
                    if count > count_final:  # Replace None in this line with the proper condition.

                        # update the final count (largest count)
                        count_final = count

                        # update the final POS
                        pos_final = pos

            # If the final POS (with the largest count) matches the true POS:
            if true_label == pos_final:  # Replace None in this line with the proper condition.
                # Update the number of correct predictions
                num_correct += 1

    accuracy = num_correct / total

    return accuracy


# create transition matrix:
def create_transition_matrix(alpha, tag_counts, transition_counts):
    '''
    Input:
        alpha: number used for smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
    Output:
        A: matrix of dimension (num_tags,num_tags)
    '''

    all_tags = sorted(tag_counts.keys())

    # number of POS tags
    num_tags = len(all_tags)

    # Initialize the transition matrix
    A = np.zeros((num_tags, num_tags))

    # Get the unique transition tuples
    trans_keys = set(transition_counts.keys())

    for i in range(num_tags):

        for j in range(num_tags):

            count = 0

            key = (all_tags[i], all_tags[j])

            if key in transition_counts:
                count = transition_counts[key]

            count_prev_tag = tag_counts.get(all_tags[i])
            A[i, j] = (count + alpha) / (count_prev_tag + (num_tags * alpha))

    return A


def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    '''
        Input:
            alpha: tuning parameter used in smoothing
            tag_counts: a dictionary mapping each tag to its respective count
            emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
            vocab: a dictionary where keys are words in vocabulary and value is an index.
                   within the function it'll be treated as a list
        Output:
            B: a matrix of dimension (num_tags, len(vocab))
    '''

    # list of all POS tags
    all_tags = sorted(tag_counts.keys())

    # number of POS tags
    num_tags = len(all_tags)

    # total number of unique words in the vocabulary
    num_words = len(vocab)

    # Initializing emission matrix B
    B = np.zeros((num_tags, num_words))

    for i in range(num_tags):
        for j in range(num_words):

            count = 0

            key = (all_tags[i], vocab[j])

            if key in emission_counts:
                count = emission_counts[key]

            count_tag = tag_counts[all_tags[i]]

            B[i, j] = (count + alpha)/(count_tag + (num_words * alpha))

    return B

# Initializing best_probs and best_paths matrices for viterbi algorithm
def initialize(states, tag_counts, A, B, corpus, vocab):
    '''
    Input:
        states: a list of all possible parts-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    '''

    num_tags = len(tag_counts)

    best_probs = np.zeros((num_tags, len(corpus)))

    best_paths = np.zeros((num_tags, len(corpus)), dtype='int')

    # define the start token
    s_idx = states.index('--s--')

    # Go through each of the POS tags
    for i in range(num_tags):  # Replace None in this line with the proper range.

        # Handle the special case when the transition from start token to POS tag i is zero
        if A[s_idx, i] == 0:  # Replace None in this line with the proper condition. # POS by word

            best_probs[i, 0] = float("-inf")

        # For all other cases when transition from start token to POS tag i is non-zero:
        else:

            best_probs[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])

    return best_probs, best_paths


def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab, verbose=True):
    '''
    Input:
        A, B: The transition and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    '''
    # Get the number of unique POS tags (which is the num of rows in best_probs)
    num_tags = best_probs.shape[0]

    # Go through every word in the corpus starting from word 1
    # Recall that word 0 was initialized in `initialize()`
    for i in range(1, len(test_corpus)):

        # Print number of words processed, every 5000 words
        if i % 5000 == 0 and verbose:
            print("Words processed: {:>8}".format(i))

        # For each unique POS tag that the current word can be
        for j in range(num_tags):  # Replace None in this line with the proper range. # for every pos tag

            # Initialize best_prob for word i to negative infinity
            best_prob_i = float("-inf")

            # Initialize best_path for current word i to None
            best_path_i = None  # Do not replace this None # @KEEPTHIS

            # For each POS tag that the previous word can be:
            for k in range(num_tags):  # Replace None in this line with the proper range.

                # Calculate the probability = None
                # best probs of POS tag k, previous word i-1 +
                # log(prob of transition from POS k to POS j) +
                # log(prob that emission of POS j is word i)
                prob = best_probs[k, i - 1] + math.log(A[k, j]) + math.log(B[j, vocab[test_corpus[i]]])

                # check if this path's probability is greater than
                # the best probability up to and before this point
                if prob > best_prob_i:  # Replace None in this line with the proper condition.

                    # Keep track of the best probability
                    best_prob_i = prob

                    # keep track of the POS tag of the previous word
                    # that is part of the best path.
                    # Save the index (integer) associated with
                    # that previous word's POS tag
                    best_path_i = k

            # Save the best probability for the
            # given current word's POS tag
            # and the position of the current word inside the corpus
            best_probs[j, i] = best_prob_i

            # Save the unique integer ID of the previous POS tag
            # into best_paths matrix, for the POS tag of the current word
            # and the position of the current word inside the corpus.
            best_paths[j, i] = best_path_i

    return best_probs, best_paths




