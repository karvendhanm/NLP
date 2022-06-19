from collections import defaultdict
from parts_of_speech_tagging.utils import *

with open('./data/WSJ_02-21.pos', 'r') as fh:
    lines = fh.readlines()

print('\t\t\tWord', '\tTag\n')
for i in range(5):
    print(f'line number {i+1}: {lines[i]}')

words = [line.split('\t')[0] for line in lines]

freq = defaultdict(int)
for word in words:
    freq[word] += 1

vocab = [key for key, value in freq.items() if (key != '\n') & (value > 1)]
vocab.sort()

# Print some random values of the vocabulary
for i in range(4000, 4005):
    print(vocab[i])

# Processing new text sources:
get_word_tag('\n', vocab)
get_word_tag('In\tIN\n', vocab)
get_word_tag('tardigrade\tNN\n', vocab)
get_word_tag('scrutinize\tVB\n', vocab)





