import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.y_hat = None 

    def forward(self, input_tensor):
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(shifted_input)
        self.y_hat = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.y_hat

    def backward(self, error_tensor):
        return self.y_hat * (error_tensor - np.sum(error_tensor * self.y_hat, axis=1, keepdims=True))