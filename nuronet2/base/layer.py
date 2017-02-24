# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 20:01:17 2017

@author: Evander
"""
from nuronet2.backend import N
from mlmodel import MLModel

class Layer(MLModel):
    def __init__(self, **kwargs):
        MLModel.__init__(self, **kwargs)
        
    """def create_input_layer(self, input_shape, input_dtype=None,
                           name=None):
        \"""
        Creates an input layer if the layer type is used without 
        one being specified.
        
        Inputs
        ------
            @param input_shape: A tuple specifying the shape (with batchsize)
                                of the layer
            @param input_dtype: Input datatype
            @param name: layer name
            
        Returns
        -------
            InputLayer instance
        \"""
        if(not name):
            prefix = self.__class__.__name__.lower() + '__input__'
            name = prefix + str(N.get_uid(prefix))
        if(not input_dtype):
            input_dtype = N.floatx
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        x = Input(shape=input_shape, dtype=input_dtype, name=name)
        return self(x)
    """