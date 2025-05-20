import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        
        # Initialize weights and bias in a single matrix
        # Add one row for bias (input_size + 1)
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
        # Store input for backward pass
        self._input_tensor = input_tensor
        
        # Add 1s column for bias
        batch_size = input_tensor.shape[0]
        ones = np.ones((batch_size, 1))
        input_with_bias = np.concatenate([input_tensor, ones], axis=1)
        
        # Store enhanced input for backward pass
        self._input_tensor = input_with_bias
        
        return np.dot(input_with_bias, self.weights)

    def backward(self, error_tensor):
        # Compute gradients
        self._gradient_weights = np.dot(self._input_tensor.T, error_tensor)
        
        # Update weights if optimizer exists
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(
                self.weights, 
                self._gradient_weights
            )
        
        # Return gradient for previous layer (excluding bias)
        return np.dot(error_tensor, self.weights.T[:, :-1])