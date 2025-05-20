import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None
        self._gradient_weights = None
        self._input_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def forward(self, input_tensor):
        self._input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1))
        input_with_bias = np.concatenate([input_tensor, ones], axis=1)
        self._input_tensor = input_with_bias
        
        return np.dot(input_with_bias, self.weights)

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(self._input_tensor.T, error_tensor)
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(
                self.weights, 
                self._gradient_weights
            )
        return np.dot(error_tensor, self.weights.T[:, :-1])