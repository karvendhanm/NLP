import pandas as pd
import math
import numpy as np

from parts_of_speech_tagging.utils import preprocess, create_dictionaries

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

# A sample of the test corpus.
print(y[:10])

_, prep = preprocess(vocab, "./data/test.words")
print('The length of the preprocessed test corpus: ', len(prep))
print('This is a sample of the test_corpus: ')
print(prep[0:10])

# Parts of speech tagging.
# build transition counts dictionary(key: (prev_tag, tag), value: number of times those tags appeared in that order),
# emission counts dictionary(key: (tag, word), value: number of times that tag & pair show up in training set)
# and tag counts dictionary(key: tag, value: number of times each tag appeared).

# write a function that takes in training corpus and returns the 3 aforementioned dictionaries.

emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)





















