import pandas as pd
from collections import defaultdict
import math
import numpy as np

with open('./data/WSJ_02-21.pos', 'r') as fh:
    training_corpus = fh.readlines()

print(training_corpus[0:5])

with open('./data/hmm_vocab.txt', 'r') as fh:
    voc_l = fh.read().split('\n')

print(voc_l[0:50])
print('few items at the end of the vocabulary list')
print(voc_l[-50:])

