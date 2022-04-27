import nltk
from nltk.corpus import twitter_samples
import random
import matplotlib.pyplot as plt
import re  # library for regular expression operations
import string  # for string operations

from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # module for stemming
from nltk.tokenize import TweetTokenizer  # module for tokenizing strings

from twitter_sentiment_analysis.utils import process_tweet

# download sample twitter dataset

nltk.download('twitter_samples')

# select the set of positive and negative tweets

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))

print('\nThe type of all_positive_tweets is: ', type(all_positive_tweets))
print('The type of a tweet entry is: ', type(all_negative_tweets[0]))

# Declare a figure with a custom size
fig = plt.figure(figsize=(5, 5))

# labels for the two classes
labels = 'Positives', 'Negative'

# Sizes for each slide
sizes = [len(all_positive_tweets), len(all_negative_tweets)]

# Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Display the chart
plt.show()

# Looking at raw texts.
# print positive in greeen
print('\033[92m' + all_positive_tweets[random.randint(0, 5000)])

# print negative in red
print('\033[91m' + all_negative_tweets[random.randint(0, 5000)])

'''
Preprocess raw text for sentiment analysis.

1) Tokenizing the string.
2) Lowercasing
3) Removing stop words and puctuation
4) Stemming

'''

# Lets pick a relatively complex tweet and try to do all preprocessing steps on it.
tweet = all_positive_tweets[2277]
print('\033[92m' + tweet)

# downloading few more libraries for preprocessing the tweet.
nltk.download('stopwords')

# Remove hyperlinks, Twitter marks and styles.
print('\033[92m' + tweet)
print('\033[94m')

# remove old style retweet text "RT"
tweet2 = re.sub(r'^RT[\s]+', '', tweet)
print(tweet2)

# remove hashtags
# only removing the hash # sign from the word
tweet2 = re.sub(r'#', '', tweet2)

print(tweet2)

# Tokenize the string
print()
print('\033[92m' + tweet2)
print('\033[94m')

# instantiate tokenizer class
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# tokenize tweets
tweet_tokens = tokenizer.tokenize(tweet2)

print()
print('Tokenized string:')
print(tweet_tokens)

# Remove stopwords and punctuations
stopwords_english = stopwords.words('english')

print('Stop words\n')
print(stopwords_english)

print('\nPunctuation\n')
print(string.punctuation)

print()
print('\033[92m')
print(tweet_tokens)
print('\033[94m')

tweets_clean = []

for word in tweet_tokens:  # Go through every word in your tokens list
    if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
        tweets_clean.append(word)

print('removed stop words and punctuation:')
print(tweets_clean)

# Stemming
# stemming is the process of converting a word to its most general form, or steam. This helps in reducing the
# size of our vocabulary.

print()
print('\033[92m')
print(tweets_clean)
print('\033[94m')

# Instantiate porter stemming class
stemmer = PorterStemmer()

# create an empty list to store the stems
tweets_stem = []

for word in tweets_clean:
    stem_word = stemmer.stem(word)  # stemming word
    tweets_stem.append(stem_word)  # append to the list

print('stemmed words:')
print(tweets_stem)

# process_tweet()

# choose the same tweet
tweet = all_positive_tweets[2277]

print()
print('\033[92m')
print(tweet)
print('\033[94m')

# call the imported function
tweet_stem = process_tweet(tweet)  # preprocess a given tweet

print('preprocessed tweet:')
print(tweets_stem)  # print the result
