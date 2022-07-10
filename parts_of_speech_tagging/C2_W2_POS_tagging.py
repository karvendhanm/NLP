import pandas as pd
import time

from parts_of_speech_tagging.utils import preprocess, create_dictionaries, predict_pos1, predict_pos2
from parts_of_speech_tagging.utils import create_transition_matrix, create_emission_matrix

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

print("transition examples")
for ex in list(transition_counts.items())[:3]:
    print(ex)

print("emission examples")
for ex in list(emission_counts.items())[:3]:
    print(ex)

for tup, cnt in emission_counts.items():
    if tup[1] == "back":
        print(tup, cnt)

states = sorted(tag_counts.keys())

# testing:
# t1 = time.time()
# for _ in range(10):
#     accuracy = predict_pos1(prep, y, emission_counts, vocab, states)
# t2 = time.time()
# for _ in range(10):
#     accuracy = predict_pos2(prep, y, emission_counts, vocab, states)
# t3 = time.time()
#
# print(f"my implementation: {t2 - t1}")
# print(f"coursera implementation: {t3 - t2}")  #Coursera's implementation is far superior, it takes ~1/60th of my time.

# Hidden Markov models for POS
# will build something context specific. will be implementing Hidden Markov Model(HMM) with a Viterbi decoder

# Generating matrices
alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)
# Testing the function
print(f"A at row 0, col 0: {A[0,0]:.9f}")
print(f"A at row 3, col 1: {A[3,1]:.4f}")

print("View a subset of transition matrix A")
A_sub = pd.DataFrame(A[30:35,30:35], index=states[30:35], columns = states[30:35] )
print(A_sub)

# create matrix 'B' emission probability matrix
alpha = 0.001
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))