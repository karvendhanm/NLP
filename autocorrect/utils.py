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
