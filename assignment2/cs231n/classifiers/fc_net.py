from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        b2 = np.zeros(num_classes)

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        X = X.reshape((-1, input_dim))
        h1, cache_h1 = affine_forward(X, self.params['W1'], self.params['b1'])
        a1, cache_a1 = relu_forward(h1)
        h2, cache_h2 = affine_forward(a1, self.params['W2'], self.params['b2'])

        scores = h2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, loss_h2 = softmax_loss(h2, y)
        loss += (self.reg * np.sum(self.params['W1'] * self.params['W1']) +
                 self.reg * np.sum(self.params['W2'] * self.params['W2'])) / 2

        da1, dW2, db2 = affine_backward(loss_h2, cache_h2)
        dW2 += self.reg * self.params['W2']

        dh1 = relu_backward(da1, cache_a1)
        dX, dW1, db1 = affine_backward(dh1, cache_h1)
        dW1 += self.reg * self.params['W1']

        grads['W1'], grads['b1'] = dW1, db1
        grads['W2'], grads['b2'] = dW2, db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.input_dim = input_dim

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pre_input = input_dim
        hidden_dims.append(num_classes)
        for i, hidden_dim in enumerate(hidden_dims):
            self.params['W%d' % (i + 1)] = \
                np.random.randn(pre_input, hidden_dim) * weight_scale
            self.params['b%d' % (i + 1)] = \
                np.zeros(hidden_dim)

            if not self.normalization is None and i < self.num_layers - 1:
                self.params['gamma%d' % (i + 1)] = \
                    np.ones((hidden_dim,))
                self.params['beta%d' % (i + 1)] = \
                    np.zeros((hidden_dim,))
            pre_input = hidden_dim

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'}
                              for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        activation_cache_ls = []
        bn_cache_ls = []
        dropout_cache_ls = []
        activation = X
        # activation = X.reshape((-1, self.input_dim))
        # print(X.shape)
        for i_minus in range(self.num_layers):
            # print(activation.shape, self.params['W%d' % (
            #     i+1)].shape, self.params['b%d' % (i+1)].shape)
            # print(i+1)
            ind = i_minus + 1
            # if ind < self.num_layers:
            # activation, cache = affine_relu_forward(
            #     activation, self.params['W%d' % (ind)], self.params['b%d' % (ind)])
            # activation_cache_ls.append(cache)

            # if self.use_dropout:
            #     activation, cache = dropout_forward(
            #         activation, self.dropout_param)
            #     dropout_cache_ls.append(cache)

            # if self.normalization == 'batchnorm':
            #     activation, cache = batchnorm_forward(
            #         activation, self.params['gamma%d' % (ind)], self.params['beta%d' % (ind)], self.bn_params[i_minus])
            #     bn_cache_ls.append(cache)

            # elif self.normalization == 'layernorm':
            #     activation, cache = layernorm_forward(
            #         activation, self.params['gamma%d' % (ind)], self.params['beta%d' % (ind)], self.bn_params[i_minus])
            #     bn_cache_ls.append(cache)
            if ind < self.num_layers:
                if not self.normalization is None:
                    activation, cache = affine_bn_relu_forward(activation, self.params['W%d' % (ind)], self.params['b%d' % (
                        ind)], self.params['gamma%d' % (ind)], self.params['beta%d' % (ind)], self.bn_params[i_minus], self.normalization)
                    bn_cache_ls.append(cache)

                else:
                    activation, cache = affine_relu_forward(
                        activation, self.params['W%d' % (ind)], self.params['b%d' % (ind)])
                    activation_cache_ls.append(cache)

                if self.use_dropout:
                    activation, cache = dropout_forward(
                        activation, self.dropout_param)
                    dropout_cache_ls.append(cache)

            else:
                output, cache = affine_forward(
                    activation, self.params['W%d' % (ind)], self.params['b%d' % (ind)])
                activation_cache_ls.append(cache)
        scores = output

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # print(scores.shape)
        loss, grad_loss = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += self.reg * np.sum(self.params['W%d' % (i+1)] ** 2) / 2

        dout = grad_loss
        grads = {}

        for inv_i in range(self.num_layers):
            ind = self.num_layers - inv_i

            if ind == self.num_layers:
                dx, dw, db = affine_backward(dout, cache)
                grads['W%d' % (ind)] = dw + self.reg * \
                    self.params['W%d' % (ind)]
                grads['b%d' % (ind)] = db
                dout = dx
            # else:
            #     if self.normalization == 'batchnorm':
            #         dout, dgamma, dbeta = batchnorm_backward_alt(
            #             dout, bn_cache_ls[ind-1])
            #         grads['gamma%d' % (ind)] = dgamma
            #         grads['beta%d' % (ind)] = dbeta

            #     elif self.normalization == 'layernorm':
            #         dout, dgamma, dbeta = layernorm_backward(
            #             dout, bn_cache_ls[ind-1])
            #         grads['gamma%d' % (ind)] = dgamma
            #         grads['beta%d' % (ind)] = dbeta

            #     if self.use_dropout:
            #         dout = dropout_backward(dout, dropout_cache_ls[ind-1])

            #     dx, dw, db = affine_relu_backward(dout, cache)
            #     grads['W%d' % (ind)] = dw + self.reg * \
            #         self.params['W%d' % (ind)]
            #     grads['b%d' % (ind)] = db
            else:
                if self.use_dropout:
                    dout = dropout_backward(dout, dropout_cache_ls[ind-1])

                if not self.normalization is None:
                    dout, dw, db, dgamma, dbeta = affine_bn_relu_backward(
                        dout, bn_cache_ls[ind-1], self.normalization)
                    grads['gamma%d' % (ind)] = dgamma
                    grads['beta%d' % (ind)] = dbeta
                    grads['W%d' % (ind)] = dw + self.reg * \
                        self.params['W%d' % (ind)]
                    grads['b%d' % (ind)] = db

                else:
                    dout, dw, db = affine_relu_backward(
                        dout, activation_cache_ls[ind-1])
                    grads['W%d' % (ind)] = dw + self.reg * \
                        self.params['W%d' % (ind)]
                    grads['b%d' % (ind)] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param, norm_kind):
    """
    Affine-bn-relu layer, a forward flow.

    @author: Xiaoke Huang, 2020/02/04

    Inputs: (x, w, b, gamma, beta, bn_param, norm_kind)

    Return tuple of: out, (affine_cache, bn_cache, relu_cache)

    """
    out, affine_cache = affine_forward(x, w, b)

    if norm_kind == 'batchnorm':
        out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
    elif norm_kind == 'layernorm':
        out, bn_cache = layernorm_forward(out, gamma, beta, bn_param)
    else:
        raise TypeError('norm kind is %s, which is wrong.' % (norm_kind))

    out, relu_cache = relu_forward(out)

    return out, (affine_cache, bn_cache, relu_cache)


def affine_bn_relu_backward(dout, cache, norm_kind):
    """
    Affine-bn-relu layer, a backward flow.

    @author: Xiaoke Huang, 2020/02/04

    Inputs: (dout, cache, norm_kind)

    Return tuple of: (dout, dw, db, dgamma, dbeta)

    """
    affine_cache, bn_cache, relu_cache = cache

    dout = relu_backward(dout, relu_cache)

    if norm_kind == 'batchnorm':
        dout, dgamma, dbeta = batchnorm_backward_alt(dout, bn_cache)
    elif norm_kind == 'layernorm':
        dout, dgamma, dbeta = layernorm_backward(dout, bn_cache)
    else:
        raise TypeError('norm kind is %s, which is wrong.' % (norm_kind))

    dout, dw, db = affine_backward(dout, affine_cache)

    return dout, dw, db, dgamma, dbeta
