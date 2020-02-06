from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, num_class = X.shape[0], W.shape[1]

    for i in range(num_train):
        scores = W.T @ X[i]
        scores -= np.max(scores)
        correct_class = y[i]
        scores = np.exp(scores)
        loss += - np.log(scores[correct_class] / np.sum(scores))

        dW[:, correct_class] += -X[i]
        for j in range(num_class):
            dW[:, j] += scores[j] / np.sum(scores) * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train, num_class = X.shape[0], W.shape[1]

    # loss
    scores = X @ W
    scores -= np.max(scores, axis=1).reshape((-1, 1))
    exp_socres = np.exp(scores)
    loss = np.sum(- np.log(exp_socres[np.arange(num_train), y] /
                           np.sum(exp_socres, axis=1)))
    loss /= num_train
    loss += reg * np.sum(W * W)

    # gradient

    # for i in range(num_train):  # ! WRONG
    #     dW[:, y[i]] += -X[i]
    weights_X = np.zeros_like(scores)
    weights_X[np.arange(num_train), y] = 1.0
    dW -= X.T @ weights_X

    dW += X.T @ (exp_socres / np.sum(exp_socres, axis=1).reshape((-1, 1)))
    dW /= num_train
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
