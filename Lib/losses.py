import numpy as np

def mse_loss(y_true, y_pred):
    m = y_true.shape[0]
    diff = y_pred - y_true
    loss = np.sum(diff**2) / m
    grad = 2 * diff / m
    return loss, grad
