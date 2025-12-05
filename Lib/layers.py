import numpy as np

class Layer:
    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, dA):
        raise NotImplementedError

    def params(self):
        return []
    

class Dense(Layer):
    def __init__(self, in_features, out_features):
        # Xavier initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        return X @ self.W + self.b

    def backward(self, dZ):
        X = self.X_cache
        m = X.shape[0]

        self.dW = (X.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        dX = dZ @ self.W.T
        return dX

    def params(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]

