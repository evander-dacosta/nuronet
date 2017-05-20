# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:24:02 2016

@author: Evander
"""

import numpy
from nuronet2.base import get_weightfactory, get_regulariser, MLModel, Layer
from nuronet2.activations import get_activation
from nuronet2.backend import N

from convlayers import normalize_tuple, normalize_padding, conv_output_length

class Pooling2d(Layer):
    def __init__(self, pool_size=(2, 2), strides=None,
                 padding='valid', **kwargs):
        super(Pooling2d, self).__init__(**kwargs)
        if(strides is None):
            strides = pool_size
        self.pool_size = normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = normalize_tuple(strides, 2, 'strides')
        self.padding = normalize_padding(padding)
        
    def _pooling_function(self, inputs, pool_size, strides, padding):
        raise NotImplementedError
        
    def build(self, input_shape):
        self.is_built = True
    
    def prop_up(self, x):
        x = self._pooling_function(inputs=x,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding)
        return x
    

        
    def get_cost(self):
        return N.cast(0.)
        
    def get_output_shape(self, input_shape):
        rows = input_shape[2]
        cols = input_shape[3]
        rows = conv_output_length(rows, self.pool_size[0],
                                  self.padding, self.strides[0])
        cols = conv_output_length(cols, self.pool_size[1],
                                  self.padding, self.strides[1])
        return (input_shape[0], input_shape[1], rows, cols)

        
class MaxPooling2d(Pooling2d):
    def __init__(self, pool_size=(2, 2), strides=None,
                 padding='valid', **kwargs):
        super(MaxPooling2d, self).__init__(pool_size, strides, padding, **kwargs)
        
    def _pooling_function(self, inputs, pool_size, strides, padding):
        output = N.pool2d(inputs, pool_size, strides, padding, pool_mode='max')
        return output

