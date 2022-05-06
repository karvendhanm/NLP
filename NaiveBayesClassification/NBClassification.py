import nltk
from nltk.corpus import twitter_samples
import numpy as np

from NaiveBayesClassification.utils import process_tweet, build_freqs

pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

# all_tweets = pos_tweets + neg_tweets
# all_tweet_labels = np.concatenate([np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))])

# 80-20 split on training and test set.
train_x = pos_tweets[:4000] + neg_tweets[:4000]
train_y = np.concatenate([np.ones(4000), np.zeros(4000)])
test_x = pos_tweets[4000:] + neg_tweets[4000:]
test_y = np.concatenate([np.ones(1000), np.zeros(1000)])

build_freqs(train_x, train_y)


