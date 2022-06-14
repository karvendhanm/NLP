import re
from collections import Counter


def words(text):
    '''

    :param text:
    :return:
    '''
    return re.findall(r'\w+', text.lower())


with open('./data/big.txt', 'r') as fh:
    big_text = fh.read()

fh.close()

WORDS = Counter(words(big_text))


def known(_words):
    '''

    :param words:
    :return:
    '''
    # The subset of 'words' that appear in the dictionary of WORDS.
    return set(w for w in _words if w in WORDS)


def candidates(word):
    '''

    :param word:
    :return:
    '''
    # Generate possible spelling corrections for a word.
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]


def Probability(word, N=sum(WORDS.values())):
    '''

    :param word:
    :param N:
    :return:
    '''
    # Probability of 'word'
    return WORDS.get(word, 0) / N


def correction(word):
    '''

    :param word:
    :return:
    '''
    # Most probable spelling correction for a word.
    return max(candidates(word), key=Probability)


def edits1(word):
    '''

    :param word:
    :return:
    '''

    # All edits that are one edit away from "word"
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    switches = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + switches + replaces + inserts)


def edits2(word):
    '''

    :param word:
    :return:
    '''

    # All edits that are two edits away from "word"
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))









