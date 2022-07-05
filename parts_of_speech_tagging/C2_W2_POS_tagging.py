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

# vocab: dictionary that has the index of the corresponding words in voc_l after it is sorted:
vocab = {}

for _idx, word in enumerate(sorted(voc_l)):
    vocab[word] = _idx

print('Vocabulary dictionary. key is the word, value is a unique integer')
cnt = 0
for k, v in vocab.items():
    print(f'{k}:{v}')
    cnt += 1
    if cnt > 20:
        break

# load in the test corpus.
with open('./data/WSJ_24.pos', 'r') as fh:
    y = fh.readlines()

print(y[:10])


