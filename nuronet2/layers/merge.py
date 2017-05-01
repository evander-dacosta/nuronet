#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 20:27:26 2017

@author: evander
"""

import numpy
from nuronet2.backend import N
from nuronet2.base import get_weightfactory, get_regulariser, Layer

class _MergeLayer(Layer):
    """
    Abstract class for elementwise merging of layers
    """
    def __init__(self, **kwargs):
        super(_MergeLayer, self).__init__(**kwargs)
    
    def _merge_function(self, inputs):
        raise NotImplementedError()
        
        
    def _elementwise_op_output_shape(self, shape1, shape2):
        if(None in [shape1, shape2]):
            return None
        elif(len(shape1) < len(shape2)):
            return self._elementwise_op_output_shape(shape2, shape1)
        elif(len(shape2) == 0):
            return shape1
        
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if(i is None or j is None):
                output_shape.append(None)
            elif(i == 1):
                output_shape.append(j)
            elif(j == 1):
                output_shape.append(i)
            else:
                if(i != j):
                    raise ValueError("Operands could not be broadcast "
                                     "together with shapes " +
                                     str(shape1) + " " + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)
            
                    
    def build(self, input_shape):
        if(not isinstance(input_shape, list)):
            raise ValueError("A merge layer should be called on a "
                             "list of inputs.")
        if(len(input_shape) < 2):
            raise ValueError("A merge layer should be called on a "
                             "list of at least 2 inputs."
                             "Got "+str(len(input_shape))+" inputs.")
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        if(len(batch_sizes) > 1):
            raise ValueError("Cannot merge tensors with different "
                             "batch sizes.")
        if(input_shape[0] is None):
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        
        for i in range(1, len(input_shape)):
            if(input_shape[i] is None):
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._elementwise_op_output_shape(output_shape,
                                                             shape)
        
        if(None not in input_shape and len(set(map(len, input_shape))) == 1):
            self._reshape_required = False
        else:
            self._reshape_required = True
        
    def prop_up(self, x):
        if(self._reshape_required):
            reshaped_inputs = []
            input_ndims = list(map(N.ndim, x))
            if(None not in input_ndims):
                max_ndim = max(input_ndims)
                for inp in x:
                    x_ndim = N.ndim(inp)
                    for _ in range(max_ndim - x_ndim):
                        inp = N.expand_dim(inp, 1)
                    reshaped_inputs.append(inp)
                return self._merge_function(reshaped_inputs)
            else:
                transposed = False
                for inp in x:
                    x_ndim = N.ndim(inp)
                    if(x_ndim is None):
                        x_shape = N.shape(inp)
                        batch_size = x_shape[0]
                        new_shape = N.concat([x_shape[1:], N.expand_dim(batch_size)])
                        x_transposed = N.reshape(inp, N.stack([batch_size, N.prod(x_shape[1:])]))
                        x_transposed = N.permute_dimensions(x_transposed, (1, 0))
                        x_transposed = N.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif(x_ndim > 1):
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(N.permute_dimensions(inp, dims))
                        transposed = True
                    else:
                        reshaped_inputs.append(inp)
                y = self._merge_function(reshaped_inputs)
                y_ndim = N.ndim(y)
                if(transposed):
                    if(y_ndim is None):
                        y_shape = N.shape(y)
                        y_ndim = N.shape(y_shape[0])
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = N.concat([N.expand_dim(batch_size),
                                              y_shape[:y_ndim - 1]])
                        y = N.reshape(y, (-1, batch_size))
                        y = N.permute_dimensions(y, dims)
                        y = N.reshape(y, new_shape)
                    elif(y_ndim > 1):
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = N.permute_dimensions(y, dims)
                return y
        else:
            return self._merge_function(inputs)
    def get_cost(self):
        return N.cast(0.)
        
    def get_output_shape(self, input_shape):
        if(input_shape[0] is None):
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if(input_shape[i] is None):
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._elementwise_op_output_shape(output_shape,
                                                             shape)
        batch_sizes = [s[0] for s in input_shape if s is not None]
        batch_sizes = set(batch_sizes)
        batch_sizes -= set([None])
        
        if(len(batch_sizes) == 1):
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape