from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import re  # library for regular expression operations
import string  # for string operations

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

            pos_cond_prob = cond_prob_dict.get(word, 0)
            if isinstance(pos_cond_prob, dict):
                pos_cond_prob = pos_cond_prob.get(str1, 0)
            neg_cond_prob = cond_prob_dict.get(word, 0)
            if isinstance(neg_cond_prob, dict):
                neg_cond_prob = neg_cond_prob.get(str2, 0)
            lambda_ += np.log((pos_cond_prob/neg_cond_prob))

        NBscore = lambda_ + prior_prob
        NBscore = float(1.0 if NBscore > 0.0 else 0.0)
        lst.append(NBscore)

    return lst

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def count_tweets(result, tweets, ys):
    '''
            Input:
                result: a dictionary that will be used to map each pair to its frequency
                tweets: a list of tweets
                ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
            Output:
                result: a dictionary mapping each pair to its frequency
    '''

    for tweet, label in zip(tweets, ys):
        for clean_word in process_tweet(tweet):

            pair = (clean_word, label)
            result[pair] = result.get(pair, 0) + 1

    return result


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

    ### START CODE HERE ###

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs.get(pair, 0)

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs.get(pair, 0)

    # Calculate D, the number of documents
    D = len(train_y)

    # Calculate D_pos, the number of positive documents
    D_pos = sum(train_y == 1)

    # Calculate D_neg, the number of negative documents
    D_neg = sum(train_y == 0)

    # Calculate logprior
    logprior = np.log((D_pos / D_neg))

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    ### END CODE HERE ###

    return logprior, loglikelihood

def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    ### START CODE HERE ###
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    ### END CODE HERE ###

    return p

def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # return this properly

    ### START CODE HERE ###
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = sum(y_hats != test_y)/len(test_x)

    # Accuracy is 1 minus the error
    accuracy = 1 - error

    ### END CODE HERE ###

    return accuracy
































