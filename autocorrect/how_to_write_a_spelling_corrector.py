import re
from collections import Counter


def words(text):
    '''

    :param text:
    :return:
    '''
    return re.findall(r'\w+', text.lower())

