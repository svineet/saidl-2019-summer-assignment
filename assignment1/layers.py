from builtins import range
import numpy as np


def affine_forward(x, w, b):
    out = None

    N = x.shape[0]
    X = x.reshape((N, -1))
    D = X.shape[1]

    out = X.dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache

    N = x.shape[0]
    X = x.reshape((N, -1))
    D = X.shape[1]

    dx, dw, db = None, None, None

    dw = X.T.dot(dout)
    dX = dout.dot(w.T)
    dx = dX.reshape(x.shape)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu(x):
    return np.maximum(np.zeros_like(x), x)


def relu_forward(x):
    out = relu(x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache

    relu_ones = relu(x)
    relu_ones[relu_ones > 0] = 1
    dx = relu_ones*dout

    return dx


def sigmoid(x):
    return np.where(x >= 0,
                    1/(1+np.exp(-x)),
                    np.exp(x)/(1+np.exp(x)))


def sigmoid_forward(x):
    out = sigmoid(x)
    cache = x, out
    return out, cache


def sigmoid_backward(dout, cache):
    x, s = cache
    dx = s*(1-s)*dout
    return dx

