import pandas as pd
from nltk.corpus import twitter_samples
import numpy as np

from NaiveBayesClassification.utils import process_tweet, build_freqs

pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

# 80-20 split on training and test set.
train_x = pos_tweets[:4000] + neg_tweets[:4000]
train_y = np.concatenate([np.ones(4000), np.zeros(4000)])
test_x = pos_tweets[4000:] + neg_tweets[4000:]
test_y = np.concatenate([np.ones(1000), np.zeros(1000)])

freq_dict = build_freqs(train_x, train_y)
# unique words in the corpus
vocabulary = list(set([key[0] for key in freq_dict.keys()]))

cols = ['word', 'pos_count', 'neg_count']
df_word_freq = pd.DataFrame(columns=cols)
# creating a dataframe with unique words in the vocabulary and count of their occurrence in positive sentiment tweets
# and negative sentiment tweets.
for word in vocabulary:
    pos_count = freq_dict.get((word, 1.0), 0)
    neg_count = freq_dict.get((word, 0.0), 0)
    arr_ = np.array([word, pos_count, neg_count]).reshape((1, -1))
    df_temp = pd.DataFrame(arr_, columns=cols)
    df_word_freq = pd.concat([df_word_freq, df_temp], axis=0)










