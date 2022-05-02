import nltk
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
from os import getcwd

from twitter_sentiment_analysis.utils import process_tweet, build_freqs

nltk.download('stopwords')
nltk.download('twitter_samples')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs)))

def sigmoid(z):
    '''

    :param z: is the input (can be a scalar or an array).
    :return: the sigmoid value of the input z. the output value ranges
    between 0 and 1, and is normally treated as a probability.
    '''

    if (isinstance(z, list)) | (isinstance(z, np.ndarray)):
        z = z[0]

    h = 1/(1+np.exp(-z))

    return h










