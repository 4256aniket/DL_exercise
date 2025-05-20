import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        
        # Add small epsilon to avoid log(0)
        epsilon = np.finfo(float).eps
        prediction_tensor = np.clip(prediction_tensor, epsilon, 1 - epsilon)
        
        # Compute cross entropy loss
        return -np.sum(label_tensor * np.log(prediction_tensor))

    def backward(self, label_tensor):
        return -(label_tensor / self.prediction_tensor)