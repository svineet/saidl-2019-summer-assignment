"""
    I have completed cs231n assignments, so I have reused the 
    code that I wrote for the affine layer part. All code is
    mine.
"""
import numpy as np
import pickle

from layer_utils import affine_relu_forward, affine_relu_backward


def mse_loss_forward(scores, y):
    return 0.5*((scores-y)**2).mean()


def mse_loss_backward(scores, y):
    # Calculate derivative wrt scores
    return (scores-y)


class BitwiseNetwork:
    def __init__(self, num_inputs, H1, H2, H3, num_outputs, std=1e-5):
        self.params = {
            'W1': np.random.randn(num_inputs, H1)*std,
            'b1': np.zeros(H1),
            'W2': np.random.randn(H1, H2)*std,
            'b2': np.zeros(H2),
            'W3': np.random.randn(H2, H3)*std,
            'b3': np.zeros(H3),
            'W4': np.random.randn(H3, num_outputs)*std,
            'b4': np.zeros(num_outputs)
        }

    def loss(self, x, y=None):
        """
            Loss function used is MSE loss
        """
        scores = None
        scores, cache1 = affine_relu_forward(x, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_relu_forward(scores, self.params['W2'], self.params['b2'])
        scores, cache3 = affine_relu_forward(scores, self.params['W3'], self.params['b3'])
        scores, cache4 = affine_relu_forward(scores, self.params['W4'], self.params['b4'])

        if y is None:
            return scores

        loss = mse_loss_forward(scores, y)

        grads = {}
        dup = mse_loss_backward(scores, y)
        dup, grads['W4'], grads['b4'] = affine_relu_backward(dup, cache4)
        dup, grads['W3'], grads['b3'] = affine_relu_backward(dup, cache3)
        dup, grads['W2'], grads['b2'] = affine_relu_backward(dup, cache2)
        dup, grads['W1'], grads['b1'] = affine_relu_backward(dup, cache1)

        return loss, grads

X = []
y = []
for d11 in range(2):
    for d12 in range(2):
        for d21 in range(2):
            for d22 in range(2):
                d1 = d11*10 + d12
                d2 = d21*10 + d22

                for op in range(2):
                    X.append([d1, d2, op])
                    b1 = d11^d21
                    b2 = d12^d22
                    if op == 0:
                        # XOR
                        y.append([b1*10 + b2])
                    else:
                        # XNOR
                        y.append([(1-b1)*10 + (1-b2)])


class RMSprop:
    def __init__(self, params, hyperparams):
        self.hyper = hyperparams
        self.params = params
        self.cache = {}

        for key, value in self.params.items():
            self.cache[key] = np.zeros_like(value)

    def step(self, loss, grads):
        decay_rate = self.hyper['rms_decay_rate'] 
        eps = self.hyper['rms_eps']
        for key in grads.keys():
            self.cache[key] = decay_rate*self.cache[key] + (1-decay_rate)*(grads[key]**2)
            final_grad = grads[key]/(np.sqrt(self.cache[key])+eps)
            model.params[key] = model.params[key] - self.hyper['lr']*final_grad

        self.hyper['lr'] *= self.hyper['lr_decay']

X = np.array(X)
y = np.array(y)

print("Training data")
print(np.hstack([X, y]))

RESUME = False
TRAIN = True
SAVE = True

LOADFILENAME = "weights_loss_8e-10.pickle"
SAVEFILENAME = "weights.pickle"

model = BitwiseNetwork(3, 500, 500, 500, 1, std=1e-2)

if RESUME:
    model.params = pickle.load(open(LOADFILENAME, "rb"))

NUM_EPOCHS = 50000
hyperparams = {
    "lr": 4e-4,
    "lr_decay": 1 - 0.000001,
    "rms_eps": 1e-8,
    "rms_decay_rate": 0.9
}
solver = RMSprop(model.params, hyperparams)
if TRAIN:
    for epoch in range(NUM_EPOCHS):
        loss, grads = model.loss(X, y)
        print("Epoch", epoch, "| Loss", loss)
        solver.step(loss, grads)

if SAVE:
    pickle.dump(model.params, open(SAVEFILENAME, "wb"))

y_pred = model.loss(X)
print("Final prediction")
print("hstack(x, y_real, y_pred)")
print(np.hstack([X, y, np.round(y_pred)]))
