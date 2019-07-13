from layers import affine_backward, affine_forward,\
                   relu_forward,    relu_backward
from layers import sigmoid_forward, sigmoid_backward


def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def affine_sigmoid_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = sigmoid_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_sigmoid_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = sigmoid_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

