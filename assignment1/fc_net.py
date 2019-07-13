import numpy as np

from layers import *

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
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
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        all_layer_sizes = [input_dim]+hidden_dims+[num_classes]
        for i, (hidden_dim, last_hidden_dim) in enumerate(zip(all_layer_sizes[1:], all_layer_sizes[:-1])):
            self.params[f'W{i+1}'] = weight_scale*np.random.randn(last_hidden_dim, hidden_dim)
            self.params[f'b{i+1}'] = weight_scale*np.zeros(hidden_dim)
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
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

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
        if self.use_batchnorm:
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
        layer_cache = []
        activations = [X]
        dropout_caches = []
        for i in range(1, self.num_layers):
            cache = None
            cur_activation, cur_cache = affine_relu_forward(activations[-1], self.params[f'W{i}'],
                                                                             self.params[f'b{i}'])

            layer_cache.append(cur_cache)
            activations.append(cur_activation)

            # after ReLU, we now add dropout layer
            # During backprop we will require the cache
            cur_drop_cache = np.nan
            if self.use_dropout:
                cur_activation, cur_drop_cache = dropout_forward(cur_activation, self.dropout_param)
                activations.append(cur_activation)
            # Append dropout cache to list anyway for alignment purposes in autistic zip statement
            # made in backprop
            dropout_caches.append(cur_drop_cache)


        # Softmax layer is done below after return scores
        scores, presoft_cache = affine_forward(activations[-1], self.params[f'W{self.num_layers}'],
                                                                self.params[f'b{self.num_layers}'])

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
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dsoftmax = softmax_loss(scores, y)
        # loss += 0.5*self.reg*sum(np.sum(thing**2) for thing in [W1, W2])
        for i in range(self.num_layers):
            loss += 0.5*self.reg*np.sum(self.params[f'W{i+1}']**2)
        
        dup, dlastW, dlastb = affine_backward(dsoftmax, presoft_cache)
        grads[f'W{self.num_layers}'] = dlastW+self.reg*self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = dlastb

        for i, cur_cache, cur_drop_cache in zip(range(self.num_layers-1, 0, -1),
                                                reversed(layer_cache),
                                                reversed(dropout_caches)):
            # Get back from dropout first
            if self.use_dropout:
                dup = dropout_backward(dup, cur_drop_cache)

            dup, dWcur, dbcur = affine_relu_backward(dup, cur_cache)
            grads[f'W{i}'] = dWcur+self.reg*self.params[f'W{i}']
            grads[f'b{i}'] = dbcur

        # Notice that dup is now dx after all the backprop till x

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
