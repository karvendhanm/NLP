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


def create_dictionaries(training_corpus):
    '''

    :param training_corpus:
    :return:
    '''
    len_training_corpus = len(training_corpus)
    transmission_count, emission_count, tag_count = {}, {}, {}
    for _idx, _elem in enumerate(training_corpus):

        if _elem.split():
            word, tag1 = _elem.split()
            tag_count[tag1] = tag_count.get(tag1, 0) + 1
            emission_count[(tag1, word)] = emission_count.get((tag1, word), 0) + 1

            if _idx == (len_training_corpus - 1):
                break

            if training_corpus[_idx + 1].split():
                _, tag2 = training_corpus[_idx + 1].split()
                transmission_count[(tag1, tag2)] = transmission_count.get((tag1, tag2), 0) + 1

    return transmission_count, emission_count, tag_count




