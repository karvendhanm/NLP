import numpy as np


def euclidean_distance(vector1, vector2):
    '''

    :param vector1:
    :param vector2:
    :return:
    '''
    return np.linalg.norm(vector1 - vector2)


def cosine_similarity(vector1, vector2):
    '''

    :param vector1:
    :param vector2:
    :return:
    '''
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return numerator / denominator


