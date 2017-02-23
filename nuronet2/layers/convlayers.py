# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:24:02 2016

@author: Evander
"""

import numpy
from nuronet2.base import get_weightfactory, get_regulariser, MLModel
from nuronet2.activations import get_activation
from nuronet2.backend import N

class Conv2dLayer(MLModel):
    def __init__(self, n, 
                 weight_factory="xavier_uniform",
                 activation="linear", border_mode='valid', strides=(1, 1),
                 w_regulariser=None,
                 b_regulariser=None, input_shape=None, **kwargs):
        """
        Arguments:
        n: A tuple (features, rows, columns)
        """
        if(border_mode not in ('valid', 'full')):
            raise Exception("Unknown border mode {}".format(border_mode) +\
            ", expected 'valid'/'full'")
        self.n_features, self.n_rows, self.n_cols = n
        self.border_mode = border_mode
        self.strides = strides
        self.weightFactory = get_weightfactory(weight_factory)
        self.activation = get_activation(activation)
        self.w_regulariser = get_regulariser(w_regulariser)
        self.b_regulariser = get_regulariser(b_regulariser)
        
        if(input_shape is not None):
            kwargs['input_shape'] = input_shape
        MLModel.__init__(self, **kwargs)
        
    @staticmethod
    def conv_shape(input_shape, filter_size, stride,
                        border_mode, pad = 0.):
        if(input_shape is None):
            return None
        if(border_mode == 'valid'):
            output_shape = input_shape - filter_size + 1
        elif(border_mode == 'full'):
            output_shape = input_shape + filter_size - 1
        output_shape = (output_shape + stride - 1) // stride
        return output_shape
        
    def prop_up(self, state):
        output = N.conv2d(state, self.W, strides=self.strides,
                          border_mode=self.border_mode,
                          filter_shape=self.W_shape)
        output += N.reshape(self.b, (1, self.n_features, 1, 1))
        return self.activation(output)
    
    def build(self, input_shape):
        if(len(input_shape) is not 4):
            raise Exception("Conv2d requires an input shape of 4." + \
            "Currently given an input_shape:{}".format(input_shape))
        stack_size = input_shape[1]
        self.W_shape = (self.n_features, stack_size, self.n_rows, self.n_cols)
        self.W = self.weightFactory(shape=self.W_shape)
        self.b = N.zeros(shape=(self.n_features,))
        self.trainable_weights = [self.W, self.b]
        self._is_built = True

    def get_cost(self):
        w_cost =  self.w_regulariser(self.W) if self.w_regulariser else N.cast(0.)
        b_cost = self.b_regulariser(self.b) if self.b_regulariser else N.cast(0.)
        return w_cost + b_cost
        
    def get_output_shape(self, input_shape):
        output_height = Conv2dLayer.conv_shape(input_shape[2], 
                                               self.n_rows,
                                               self.strides[0],
                                               self.border_mode)
        output_width = Conv2dLayer.conv_shape(input_shape[3], 
                                               self.n_cols,
                                               self.strides[1],
                                               self.border_mode)
        return (input_shape[0], self.n_features, output_height, output_width)
        
        
class Maxpool2d(MLModel):
    def __init__(self, pool_size=(2, 2), strides=(1, 1), 
                 ignore_border=False, pad=(0, 0), **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        self.ignore_border = ignore_border
        self.pad = pad
        MLModel.__init__(self, **kwargs)
        
    @staticmethod
    def pool_shape(inputShape, poolSize, stride, 
                        ignore_border, pad):
        if(inputShape is None or poolSize is None):
            return None
        if(ignore_border):
            outputShape = inputShape + (2 * pad) - poolSize + 1
            outputShape = (outputShape + stride - 1) // stride
            
        else:
            assert pad == 0.
            if(stride >= poolSize):
                outputShape = (inputShape + stride - 1) // stride
            else:
                outputShape = max(0, (inputShape - poolSize + stride - 1) // stride) + 1
            
        return outputShape

    def prop_up(self, state):
        output = N.pool2d(state, self.pool_size, self.strides,
                          self.pad, self.ignore_border,
                          pool_mode='max')
        return output
        
    def build(self, input_shape):
        if(len(input_shape) is not 4):
            raise Exception("Maxpool2d requires an input shape of 4." + \
            "Currently given an input_shape:{}".format(self.input_shape))
        self._is_built = True
        
    def get_cost(self):
        return N.cast(0.)

    def get_output_shape(self, input_shape):
        output_height = Maxpool2d.pool_shape(input_shape[2],
                                             self.pool_size[0],
                                            self.strides[0],
                                            self.ignore_border, self.pad[0])
        output_width = Maxpool2d.pool_shape(input_shape[3],
                                             self.pool_size[1],
                                            self.strides[1],
                                            self.ignore_border, self.pad[1])
        return (input_shape[0], input_shape[1], output_height, 
                output_width)
        
        

        
if __name__ == "__main__":
    from nuronet2.layers import *
    
    layer1 = Input((1, 32,32))
    layer2 = Conv2dLayer((5, 2, 2))
    
    out = layer2(layer1)