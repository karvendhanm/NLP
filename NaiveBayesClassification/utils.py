import numpy as np
import re  # library for regular expression operations
import string  # for string operations

from collections import defaultdict

import pandas as pd
from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # module for stemming
from nltk.tokenize import TweetTokenizer  # module for tokenizing strings


def process_tweet(tweet):
    '''

    :param tweet:
    :return:
    '''

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, labels):
    '''

    :param tweets:
    :param labels:
    :return:
    '''

    freq_dict = {}
    for tweet, label in zip(tweets, labels):
        for word in process_tweet(tweet):
            pair = (word, label)
            freq_dict[pair] = freq_dict.get(pair, 0) + 1.0

    return freq_dict


def get_naive_bayes_score(tweets, cond_prob_dict, n_pos_tweets, n_neg_tweets,
                       str1='pos_conditional_prob', str2='neg_conditional_prob'):
    '''
    calcualtes Naive Bayes classification for all input tweets. 1 is positive and 0 is negative
    :param tweets:
    :param cond_prob_dict:
    :param n_pos_tweets:
    :param n_neg_tweets:
    :param str1:
    :param str2:
    :return:
    '''

    lst = []
    prior_prob = np.log((n_pos_tweets/n_neg_tweets))
    for tweet in tweets:
        lambda_ = 0.0
        for word in process_tweet(tweet):

            pos_cond_prob = cond_prob_dict.get(word, 0.0000001)
            if isinstance(pos_cond_prob, dict):
                pos_cond_prob = pos_cond_prob.get(str1, 0.0000001)
            neg_cond_prob = cond_prob_dict.get(word, 0.0000001)
            if isinstance(neg_cond_prob, dict):
                neg_cond_prob = neg_cond_prob.get(str2, 0.0000001)
            lambda_ += np.log((pos_cond_prob/neg_cond_prob))

        NBscore = lambda_ + prior_prob
        NBscore = float(1.0 if NBscore >= 1.0 else 0.0)
        lst.append(NBscore)

    return lst

