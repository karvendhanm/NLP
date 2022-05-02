import numpy as np


def sigmoid(z):
    '''

    :param z: is the input (can be a scalar or an array).
    :return: the sigmoid value of the input z. the output value ranges
    between 0 and 1, and is normally treated as a probability.
    '''

    # if (isinstance(z, list)) | (isinstance(z, np.ndarray)):
    #     z = z[0]

    h = 1 / (1 + np.exp(-z))

    h = np.array(h).reshape(-1, 1)

    return h


def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    ### START CODE HERE ###
    # get 'm', the number of rows in matrix x
    m = x.shape[0]

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = -(np.dot(np.transpose(y), np.log(h)) + np.dot(np.transpose(1 - y), np.log(1 - h))) / m
        print(f'cost on iteration {i} is: {J}')

        # update the weights theta
        theta = theta - ((alpha / m) * np.dot(np.transpose(x), (h - y)))

    ### END CODE HERE ###
    J = float(J)
    return J, theta


# Check the function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

# Apply gradient descent
tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")



















