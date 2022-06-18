from collections import Counter
import re

def process_data(file_name):
    # converts a text to lowercase and returns a list of words.
    '''

    :param file_name:
    :return:
    '''

    regex_pattern = r'\w+'

    words = []
    with open(file_name, 'r') as fh:
        line = fh.readline()
        while line:
            line = line.lower()
            match = re.findall(regex_pattern, line)
            words.extend(match)
            line = fh.readline()

    return words


def get_count(word_l):
    '''
    Input:
        word_l: a set of words representing the corpus.
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    '''

    word_count_dict = Counter(word_l)
    return word_count_dict


def get_probs(word_count_dict):
    '''
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur.
    '''
    probs = {}

    # get the total count of words for all words in the dictionary
    tot_words = sum(word_count_dict.values())
    for key, value in word_count_dict.items():
        probs[key] = (value / tot_words)

    return probs


def delete_letter(word, verbose=False):
    '''
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''

    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    delete_l = [L + R[1:] for L, R in split_l if R]

    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l


def switch_letter(word, verbose=False):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    '''

    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    switch_l = [L + R[1] + R[0] + R[2:] for L, R in split_l if len(R) >= 2]

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l


def replace_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    '''

    letters = 'abcdefghijklmnopqrstuvwxyz'

    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    replace_set = [L + letter + R[1:] for L, R in split_l if R for letter in letters]
    replace_set = set(filter(lambda x: x != word, replace_set))

    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l


def insert_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []

    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    insert_l = [L + letter + R for L, R in split_l for letter in letters]

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l


# Edit one letter
def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible words that are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """
    edit_one_set = delete_letter(word) + replace_letter(word) + insert_letter(word)

    if allow_switches:
        edit_one_set = edit_one_set + switch_letter(word)

    return set(edit_one_set)


# Edit two letters
def edit_two_letters(word, allow_switches=True):
    '''
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''

    if allow_switches:
        edit_two_set = [w2 for w1 in edit_one_letter(word) for w2 in edit_one_letter(w1)]
    else:
        edit_two_set = [w2 for w1 in edit_one_letter(word, False) for w2 in edit_one_letter(w1, False)]

    # return this as a set instead of a list
    return set(edit_two_set)


def get_corrections(word, probs, vocab, n=2, verbose=False):
    '''
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''

    suggestions = list(
        set([w for w in vocab if w == word]) or edit_one_letter(word) or edit_two_letters(word) or set(word))

    # Step 2: determine probability of suggestions

    best_words = {}
    for word in suggestions:
        best_words[word] = probs.get(word, 0)

    # Step 3: Get all your best words and return the most probable top n_suggested words as n_best

    n_best = list(sorted(best_words.items(), key=lambda item: item[1], reverse=True)[:n])

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


