# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:01:17 2017

@author: Evander
"""
from nuronet2.backend import N
from mlmodel import MLModel, MLConnection

class Layer(MLModel):
    def __init__(self, **kwargs):
        MLModel.__init__(self, **kwargs)
            
        

class InputLayer(Layer):
    def __init__(self, input_shape, input_dtype=N.floatx, input_tensor=None,
                 name=None):
        if(input_tensor is None):
            input_tensor = N.variable(ndim=len((input_shape)) + 1
                                ,dtype=input_dtype, name=name)
            input_tensor._nuro_shape = (None,) + input_shape
        else:
            input_shape = input_tensor._shape
        input_tensor._nuro_history = (self, 0, 0)
        Layer.__init__(self, input_shape=input_shape, 
                                    input_dtype=input_dtype,
                                    name=name)
        MLConnection(self, inbound_models=[], connection_indices=[],
                     tensor_indices=[], input_tensors=[input_tensor],
                     output_tensors=[input_tensor],
                     input_shapes=[self.batch_input_shape],
                     output_shapes=[self.batch_input_shape])
                    
    def build(self, input_shape):
        self.is_built = True
        
    def prop_up(self, x):
        return x
        
    def get_cost(self):
        return N.cast(0.)
        
    def get_output_shape(self, input_shape):
        return self.batch_input_shape
        

            
def Input(shape=None, name=None, dtype=N.floatx,
                      tensor=None):
    """Used to instantiate a Nuronet tensor that is augmented with
    _nuro_shape and _nuro_history attributes.
    
    These attributes allow us to build models by just specifying the input
    and output tensors without the underlying model/layer connections.
    """
    if not shape and tensor is None:
        assert shape, ("Input() requires a shape argument")
    input_layer = InputLayer(input_shape=tuple(shape),
                             name=name, input_dtype=dtype,
                             input_tensor=tensor)
    outputs = input_layer.inbound_connections[0].output_tensors
    if(len(outputs) == 1):
        return outputs[0]
    else:
        return outputs