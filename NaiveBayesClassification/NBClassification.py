import pandas as pd
from nltk.corpus import twitter_samples
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from NaiveBayesClassification.utils import process_tweet, build_freqs, get_naive_bayes_score

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

df_word_freq['pos_count'] = df_word_freq['pos_count'].astype('float')
df_word_freq['neg_count'] = df_word_freq['neg_count'].astype('float')

# The number of unique words in the vocabulary should be equal to the number of rows in the dataframe 'df_word_freq'.
assert len(vocabulary) == len(df_word_freq), 'issue with the number of unique words in vocabulary'

# Creating conditional probability of each unique word in the vocabulary given that the word has occurred in
# positive tweet or in negative tweet.
nPosWords = sum(df_word_freq['pos_count'])
nNegWords = sum(df_word_freq['neg_count'])
nUniqueWords = len(df_word_freq)

df_word_freq['pos_conditional_prob'] = df_word_freq.apply(lambda x:
                                                          (x['pos_count'] + 1)/(nPosWords + nUniqueWords), axis=1)
df_word_freq['neg_conditional_prob'] = df_word_freq.apply(lambda x:
                                                          (x['neg_count'] + 1)/(nNegWords + nUniqueWords), axis=1)
df_cond_prob = df_word_freq.drop(labels=['pos_count', 'neg_count'], axis=1)

# creating a lookup dictionary that has a pos(+) and neg(-) conditional probability of each unique word in vocabulary,
dict_cond_prob = df_cond_prob.set_index('word').to_dict('index')

# Naive Bayes classification
lst_ = get_naive_bayes_score(test_x, dict_cond_prob, len(pos_tweets[:4000]), len(neg_tweets[:4000]))

acc_ = accuracy_score(test_y, lst_)
precision_ = precision_score(test_y, lst_)
recall_ = recall_score(test_y, lst_)

print(f'the accuracy is: {acc_}, the precision is: {precision_}, the recall is: {recall_}')















