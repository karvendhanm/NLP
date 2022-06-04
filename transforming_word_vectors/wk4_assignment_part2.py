# Naive Machine translation and LSH

import pickle
import string

import time

import nltk
import numpy as np
from nltk.corpus import stopwords, twitter_samples

from transforming_word_vectors.utils_nb import (get_dict, process_tweet, cosine_similarity, get_matrices, compute_loss)
from transforming_word_vectors.utils_nb import compute_gradient, align_embeddings, nearest_neighbor, test_vocabulary
from transforming_word_vectors.utils_nb import get_document_embedding, get_document_vecs, hash_value_of_vector
from transforming_word_vectors.utils_nb import make_hash_table

en_embeddings_subset = pickle.load(open('./data/Files/en_embeddings.p', 'rb'))
fr_embeddings_subset = pickle.load(open('./data/Files/fr_embeddings.p', 'rb'))

# get the positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets

custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
tweet_embedding = get_document_embedding(custom_tweet, en_embeddings_subset)
tweet_embedding[-5:]

document_vecs, ind2Tweet = get_document_vecs(all_tweets, en_embeddings_subset)

# looking up the tweets:
my_tweet = 'i am sad'
tweet_embedding = get_document_embedding(my_tweet, en_embeddings_subset)
cosine_similarity(document_vecs, tweet_embedding)

print(cosine_similarity(document_vecs, tweet_embedding))
idx = np.argmax(cosine_similarity(document_vecs, tweet_embedding))
print(all_tweets[idx])

N_VECS = len(all_tweets)       # This many vectors.
N_DIMS = len(ind2Tweet[1])     # Vector dimensionality.
print(f"Number of vectors is {N_VECS} and each has {N_DIMS} dimensions.")

# The number of planes. We use log2(256) to have ~16 vectors/bucket.
N_PLANES = 10
# Number of times to repeat the hashing to improve the search.
N_UNIVERSES = 25

np.random.seed(0)
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES))
            for _ in range(N_UNIVERSES)]

np.random.seed(0)
idx = 0
planes = planes_l[idx]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
hash_value_of_vector(vec, planes)

planes = planes_l[0]  # get one 'universe' of planes to test the function
tmp_hash_table, tmp_id_table = make_hash_table(document_vecs, planes)

print('just for debugging')










