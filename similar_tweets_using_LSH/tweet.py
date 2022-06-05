import numpy as np
import pandas as pd
import pickle

from transforming_word_vectors.utils_nb import process_tweet

from nltk.corpus import twitter_samples

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets

en_embeddings_subset = pickle.load(open('./data/Files/en_embeddings.p', 'rb'))












