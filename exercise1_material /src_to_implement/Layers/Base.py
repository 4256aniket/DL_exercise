class BaseLayer:
    def __init__(self):
        self.trainable = False
        self._optimizer = None
    
    @property                    
    def optimizer(self):        
        return self._optimizer
    
    @optimizer.setter           
    def optimizer(self, value): 
        self._optimizer = value