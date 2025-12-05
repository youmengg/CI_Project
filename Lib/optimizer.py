class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def step(self, params):
        for param, grad in params:
            param -= self.lr * grad
