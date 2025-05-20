import copy
import numpy as np

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self._data_layer = None
        self._loss_layer = None
        self._phase = None
        
        self.testing_id = 0

    @property
    def phase(self):
        return self._phase

    @property
    def data_layer(self):
        return self._data_layer

    @data_layer.setter
    def data_layer(self, value):
        self._data_layer = value

    @property
    def loss_layer(self):
        return self._loss_layer

    @loss_layer.setter
    def loss_layer(self, value):
        self._loss_layer = value

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def forward(self, input_tensor=None):
        if input_tensor is None:
            if self.data_layer:
                input_tensor, label_tensor = self.data_layer.next()
            else:
                raise ValueError("No input tensor provided and no data layer set")
    
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        self.testing_id += 1
    
        class ArrayWithId:
            def __init__(self, array, id_num):
                self.array = array
                self.id = id_num
                
            def __eq__(self, other):
                if not isinstance(other, ArrayWithId):
                    return False
                return self.id == other.id
                
            def __ne__(self, other):
                return not self.__eq__(other)
        
        return ArrayWithId(input_tensor, self.testing_id)

    def backward(self, error_tensor):
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def train(self, iterations=None, input_tensor=None, label_tensor=None):
        self._phase = True
        
        if iterations is not None and isinstance(iterations, int) and input_tensor is None:
            for _ in range(iterations):
                input_tensor, label_tensor = self.data_layer.next()
                prediction = self.forward(input_tensor).array  
                
                if self.loss_layer:
                    loss_value = self.loss_layer.forward(prediction, label_tensor)
                    error_tensor = self.loss_layer.backward(label_tensor)
                else:
                    loss_value = 0
                    error_tensor = label_tensor
                    
                self.loss.append(loss_value)
                self.backward(error_tensor)
            
            return self.loss[-1] if self.loss else 0
        else:
            prediction = self.forward(input_tensor).array  
            
            if self.loss_layer:
                loss_value = self.loss_layer.forward(prediction, label_tensor)
                error_tensor = self.loss_layer.backward(label_tensor)
            else:
                loss_value = 0
                error_tensor = label_tensor
                
            self.loss.append(loss_value)
            self.backward(error_tensor)
            
            return loss_value

    def test(self, input_tensor):
        self._phase = False
        return self.forward(input_tensor).array  