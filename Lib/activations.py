import numpy as np
from lib.layers import Layer

class ReLU(Layer):
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask
    
    def backward(self, dA):
        return dA * self.mask
    

class Sigmoid(Layer):
    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out
    
    def backward(self, dA):
        return dA * (self.out * (1 - self.out))


class Tanh(Layer):
    def forward(self, X):
        self.out = np.tanh(X)
        return self.out
    
    def backward(self, dA):
        return dA * (1 - self.out**2)
