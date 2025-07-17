import numpy as np
from Optimization import Optimizers
from Layers import Base
import copy

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(size=(input_size + 1, output_size))
        self.gradient_weights = None
        self._optimizer = None
        self.temp = []

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        self.lastIn = input_tensor
        self.augmented_input = np.concatenate([input_tensor, np.ones((input_tensor.shape[0], 1))], axis=1)
        self.lastOut = np.dot(self.augmented_input, self.weights)
        return self.lastOut
     
    def backward(self, error_tensor):
        dx = np.dot(error_tensor, self.weights[:-1, :].T)
        dW = np.dot(self.augmented_input.T, error_tensor)
        
        if self._optimizer is not None:
            self.weights = self._optimizer.weight.calculate_update(self.weights, dW)
        
        self.gradient_weights = dW
        return dx

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize(
            (self.input_size, self.output_size), 
            self.input_size, 
            self.output_size
        )
        
        bias = bias_initializer.initialize(
            (1, self.output_size), 
            1, 
            self.output_size
        )
        self.weights = np.vstack([weights, bias])