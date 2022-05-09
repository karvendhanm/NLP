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

