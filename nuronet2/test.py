import warnings
from collections import OrderedDict
from nuronet2.backend import N
from mlmodel import MLModel, MLConnection, make_list

class NeuralNetwork(MLModel):
    def __init__(self, layers=None, name=None, **kwargs):
        self.layers = []
        self.model = None
        self.inputs = []
        self.outputs = []
        self._trainable = True #required?
        
        super(MLModel, self).__init__(**kwargs)
        
        if(layers):
            for layer in layers:
                self.add(layer)
                
    def add(self, layer):
        """
        Adds a layer to the neural network's layer stack
        
        Inputs
        ------
            @param layer : A layer instance
        """
        if(not is)
        
    
        
        
