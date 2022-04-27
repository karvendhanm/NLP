import nltk
from os import getcwd
import pandas as pd
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np

from twitter_sentiment_analysis.utils import build_freqs, process_tweet
nltk.download('twitter_samples')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = all_positive_tweets + all_negative_tweets
labels = np.append(np.ones((len(all_positive_tweets), 1)), np.zeros((len(all_negative_tweets), 1)), axis=0)

train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
print("Number of tweets: ", len(train_x))
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))

freqs = build_freqs(train_x, train_y)
# data = pd.read_csv('./data/logistic_features.csv')

tweet_pos_neg_cnt = []
for idx, tweet in enumerate(train_x):
    # cleaned tweet, stopwords removed, words stemmed
    pos_cnt = 0
    neg_cnt = 0
    for word in process_tweet(tweet):
        pos_cnt += freqs.get((word, 1), 0)
        neg_cnt += freqs.get((word, 0), 0)
    tweet_pos_neg_cnt.append([idx, pos_cnt, neg_cnt])


print('just for debugging')
