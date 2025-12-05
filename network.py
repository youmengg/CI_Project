from lib.layers import Layer
from lib.optimizer import SGD

class Network:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def params(self):
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.params())
        return all_params
    
    def fit(self, X, y, loss_fn, epochs=2000):
        losses = []
        for epoch in range(epochs):
            # Forward
            y_pred = self.forward(X)
            loss, grad = loss_fn(y, y_pred)

            # Backward
            self.backward(grad)

            # Update
            if self.optimizer:
                self.optimizer.step(self.params())

            losses.append(loss)

            if epoch % 200 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")

        return losses
