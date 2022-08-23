from auto_complete.utils import *

n_gram_counts = {
    ('i', 'am', 'happy'): 2,
    ('am', 'happy', 'because'): 1}

# get count of n-gram tuple.
print(f"count of n-gram ('i', 'am', 'happy'): {n_gram_counts[('i', 'am', 'happy')]}")

if ('i', 'am', 'learning') in n_gram_counts:
    print(f"n-gram {('i', 'am', 'learning')} found")
else:
    print(f"n-gram {('i', 'am', 'learning')} missing")

# update the count in the word count dictionary
n_gram_counts[('i', 'am', 'learning')] = 1
if ('i', 'am', 'learning') in n_gram_counts:
    print(f"n-gram {('i', 'am', 'learning')} found")
else:
    print(f"n-gram {('i', 'am', 'learning')} missing")

# merging two tuples in python
prefix = ('i', 'am', 'happy')
word = 'because'
n_gram = prefix + (word,)
print(n_gram)

# building a count matrix
corpus = ['I', 'am', 'happy', 'because', 'I', 'am', 'learning', '.']
bigram, vocabulary, count_matrix = single_pass_trigram_count_matrix(corpus)

# Buildling a probability matrix
row_sums = count_matrix.sum(axis=1)
# divide each row by its sum
prob_matrix = count_matrix.div(row_sums, axis=0)











