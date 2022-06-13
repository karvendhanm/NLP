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

