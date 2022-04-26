import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt              # visualization library
import numpy as np

from twitter_sentiment_analysis.utils import process_tweet, build_freqs

nltk.download('twitter_samples')
nltk.download('stopwords')

# select the lists of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = all_positive_tweets + all_negative_tweets

# let's see how many tweets we have
print("Number of tweets: ", len(tweets))

# make a numpy array representing labels of the tweets
labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

# create frequency dictionary
freqs = build_freqs(tweets, labels)

# check data type
print(f'type(freqs) = {type(freqs)}')

# check length of the dictionary
print(f'len(freqs) = {len(freqs)}')

#Table of word counts
# select some words to appear in the report. we will assume that each word is unique (i.e. no duplicates)
keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']

data = []
for word in keys:

    pos_cnt = freqs.get((word, 1), 0)
    neg_cnt = freqs.get((word, 0), 0)

    data.append([word, pos_cnt, neg_cnt])

fig, ax = plt.subplots(figsize=(8,8))

x = np.log([x[1]+1 for x in data])
y = np.log([x[2]+1 for x in data])

ax.scatter(x, y)
plt.xlabel('Log positive count')
plt.ylabel('Log negative count')

for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([0, 9], [0, 9], color='red')

plt.show()