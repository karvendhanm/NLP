import nltk
from nltk.corpus import twitter_samples
import random
import matplotlib.pyplot as plt
import re  # library for regular expression operations
import string  # for string operations

from collections import defaultdict
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









