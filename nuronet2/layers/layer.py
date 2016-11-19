# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 00:36:57 2016

@author: evander
"""

from nuronet2.base import  MLModel, InputDetail, MLConnection
from nuronet2.activations import get_activation
from nuronet2.base import get_weightfactory, get_regulariser
from nuronet2.backend import N

class InputLayer(MLModel):
    def __init__(self, input_shape, input_dtype=N.floatx, input_tensor=None,
                 name=None):
        if(input_tensor is None):
            input_tensor = N.variable(ndim=len((input_shape)) + 1
                                ,dtype=input_dtype, name=name)
            input_tensor._nuro_shape = (None, input_shape[0])
        else:
            input_shape = input_tensor._shape
        input_tensor._nuro_history = (self, 0, 0)
        MLModel.__init__(self, input_shape=input_shape, input_dtype=input_dtype,
                         name=name)
        MLConnection(self, inbound_models=[], connection_indices=[],
                     tensor_indices=[], input_tensors=[input_tensor],
                    output_tensors=[input_tensor], input_shapes=[self.input_shape],
                    output_shapes=[self.input_shape])
                    
    def build(self, input_shape):
        self.is_built = True
        
    def prop_up(self, x):
        return x
        
    def get_cost(self):
        return N.cast(0.)
        
    def get_output_shape(self, input_shape):
        return input_shape
        

            
def Input(shape=None, name=None, dtype=N.floatx,
                      tensor=None):
    """Used to instantiate a Nuronet tensor that is augmented with
    _nuro_shape and _nuro_history attributes.
    
    These attributes allow us to build models by just specifying the input
    and output tensors without the underlying model/layer connections.
    """
    input_layer = InputLayer(input_shape=tuple(shape),
                             name=name, input_dtype=dtype,
                             input_tensor=tensor)
    outputs = input_layer.inbound_connections[0].output_tensors
    if(len(outputs) == 1):
        return outputs[0]
    else:
        return outputs
        
        
class DenseLayer(MLModel):
    def __init__(self, n, weight_factory='xavier_uniform',
                 activation='linear', weights=None, w_regulariser=None,
                 b_regulariser=None, input_shape=None, **kwargs):
        self.weightFactory = get_weightfactory(weight_factory)
        self.activation = get_activation(activation)
        self.w_regulariser = get_regulariser(w_regulariser)
        self.b_regulariser = get_regulariser(b_regulariser)
        
        self.n = n
        self.input_details = [InputDetail(ndim=2)]
        if(input_shape is not None):
            kwargs['input_shape'] = input_shape
        MLModel.__init__(self, **kwargs)
        
    def build(self, input_shape):
        assert(len(input_shape) == 2)
        input_dim = input_shape[1]
        self.input_details = [InputDetail(dtype=N.floatx, shape=(None, input_dim))]
        self.W = self.weightFactory(shape=(input_shape[-1], self.n))
        self.b = N.zeros(shape=(self.n,))
        self.trainable_weights = [self.W, self.b]
        self._is_built = True
        
    def prop_up(self, x):
        return self.activation(N.dot(x, self.W) + self.b)
    
    def get_cost(self):
        w_cost =  self.w_regulariser(self.W) if self.w_regulariser else N.cast(0.)
        b_cost = self.b_regulariser(self.b) if self.b_regulariser else N.cast(0.)
        return w_cost + b_cost
            
    def get_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.n)
            
            
        
    
        
        
if __name__ == "__main__":
    a = Input((3,), name="a")
    b = DenseLayer(2)
    y = b(a)
    f = N.function([a], y)