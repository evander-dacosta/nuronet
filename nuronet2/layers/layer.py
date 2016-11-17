# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 00:36:57 2016

@author: evander
"""
from collections import OrderedDict
from nuronet2.base import MLModel
from backend import N

class Layer(MLModel):
    def __init__(self, input_shape, name=None):
        MLModel.__init__(self, input_shape=input_shape, name=name)
        
    def set_params(self, params):
        if(not isinstance(params, (list, tuple))):
            raise Exception("Parameters set to a layer must be list/tuple")
        currentParams = self.get_params()
        for oldParam, newParam in zip(currentParams, params):
            oldShape = N.get_value(oldParam).shape#oldParam.get_value().shape
            newShape = newParam.shape
            oldShapes = 'x'.join(map(str, oldShape))
            newShapes = 'x'.join(map(str, newShape))
            
            if(oldShapes == newShapes):
                N.set_value(oldParam, newParam.astype('float32'))
            else:
                raise Exception("Mismatching shapes for current layer:" + \
                " meant to be {}, given {}".format(oldShapes, newShapes))
                
    #To be Implemented
    def prop_up(self, *args, **kwargs):
        raise NotImplementedError()
        
    def build(self, *args, **kwargs):
        raise NotImplementedError()
        
    def get_cost(self):
        raise NotImplementedError()
        
    def get_params(self):
        raise NotImplementedError()
        
    def get_output_shape(self):
        raise NotImplementedError()
        
    def get_updates(self):
        """
        Return custom updates
        """
        return OrderedDict()
                
                
                
class InputLayer(Layer):
    def __init__(self, n, input_shape=None, name=None):
        #regular inputs
        if(isinstance(n, int)):
            input_shape = (None, n) if input_shape is None else input_shape
            n = (n,)
            
        #convolutional inputs always specified as (n_channels, n_rows, n_cols)
        elif(isinstance(n, (list, tuple))):
            n = tuple(n)
            if(n[0] is None):
                n = n[1:]
            if(input_shape is None):
                input_shape=[input_shape]
                input_shape.extend(n)
            input_shape = tuple(input_shape)
        
        if(not tuple(input_shape[1:]) == n):
            raise Exception("Input shape is mismatched with n." + \
                    "Given n:{}. Given input_shape:{}".format(n, input_shape))
        self.n = n
        Layer.__init__(self, input_shape=input_shape, name=name)
        self.build()
        
    def prop_up(self, state):
        print "{}'s prop_up called".format(self.name)
        return state        
        
    def build(self, *args, **kwargs):
        self._is_built = True
        
    def get_cost(self):
        return N.cast(0.)
        
    def get_params(self):
        return []
        
    def get_output_shape(self):        
        return self.input_shape
                
                
if __name__ == "__main__":
    x = InputLayer(2, input_shape=(None, 2), name='x')
    y = InputLayer(2, input_shape=(None, 2), name='y')
    a = x(y)