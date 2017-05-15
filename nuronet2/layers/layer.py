# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 00:36:57 2016

@author: evander
"""
import numpy
from nuronet2.base import InputDetail
from nuronet2.activations import get_activation
from nuronet2.backend import N
from nuronet2.base import get_weightfactory, get_regulariser, Layer


        
        
class DenseLayer(Layer):
    def __init__(self, n, weight_factory='xavier_uniform',
                 activation='linear', weights=None, w_regulariser=None,
                 b_regulariser=None, input_shape=None,
                 **kwargs):
        self.weightFactory = get_weightfactory(weight_factory)
        self.activation = get_activation(activation)
        self.w_regulariser = get_regulariser(w_regulariser)
        self.b_regulariser = get_regulariser(b_regulariser)
        
        if(input_shape is not None):
            self.input_dim = input_shape[-1]
        else:
            self.input_dim = None
        
        self.n = n
        if(input_shape is not None):
            kwargs['input_shape'] = input_shape
        Layer.__init__(self, **kwargs)
        
        
    def build(self, input_shape):
        assert(len(input_shape) >= 2)
        input_dim = input_shape[-1]
        self.input_details = [InputDetail(dtype=N.floatx, ndim=len(input_shape))]
        
        self.W = self.weightFactory(shape=(input_dim, self.n))
        self.b = N.zeros(shape=(self.n,))
        self.trainable_weights = [self.W, self.b]
        self.is_built = True
        
    def prop_up(self, x):
        return self.activation(N.dot(x, self.W) + self.b)
    
    def get_cost(self):
        w_cost =  self.w_regulariser(self.W) if self.w_regulariser else N.cast(0.)
        b_cost = self.b_regulariser(self.b) if self.b_regulariser else N.cast(0.)
        return w_cost + b_cost
            
    def get_output_shape(self, input_shape):
        assert input_shape and isinstance(input_shape, tuple)
        assert len(input_shape) >= 2
        assert input_shape[-1]
        if(self.input_dim):
             assert input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.n
        return tuple(output_shape)
        
        
class Flatten(Layer):
    def __init__(self, input_shape=None, **kwargs):
        if(input_shape is not None):
            kwargs['input_shape'] = input_shape
        Layer.__init__(self, **kwargs)

        
    def prop_up(self, state):
        input_shape = state._nuro_shape
        state = N.batch_flatten(state)
        state._nuro_shape = self.get_output_shape(input_shape)
        return state
    
    def build(self, input_shape):
        self.is_built = True
        
    def get_cost(self):
        return N.cast(0.)
        
    def get_output_shape(self, input_shape):
        return (input_shape[0], numpy.prod(input_shape[1:]))



class Activation(Layer):
    def __init__(self, activation, **kwargs):
        self.activation = get_activation(activation)
        Layer.__init__(self, **kwargs)        
        
    def build(self, input_shape):
        self.is_built = True
        
    def prop_up(self, x):
        return self.activation(x)
        
    def get_cost(self):
        return N.cast(0.)
        
    def get_output_shape(self, input_shape):
        return input_shape

      
class Dropout(Layer):
    def __init__(self, p, **kwargs):
        assert 0. < p < 1.
        self.p = p
        Layer.__init__(self, **kwargs)
    
    def build(self, input_shape):
        self.is_built = True
        
    def prop_up(self, x):
        if(self.is_training):
            x = N.dropout(x, self.p)
        return x
        
    def get_cost(self):
        return N.cast(0.)
        
    def get_output_shape(self, input_shape):
        return input_shape

            
            
        
    
        
        
if __name__ == "__main__":
    a = Input((3,), name="a")
    b = DenseLayer(2)
    y = b(a)
    f = N.function([a], y)