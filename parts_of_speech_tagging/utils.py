from collections import defaultdict
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
        emission_counts[(word, tag)] += 1
        tag_counts[tag] += 1

        prev_tag = tag

    return emission_counts, transition_counts, tag_counts








