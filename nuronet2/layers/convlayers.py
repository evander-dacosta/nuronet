# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:24:02 2016

@author: Evander
"""

import numpy
from nuronet2.base import get_weightfactory, get_regulariser, MLModel, Layer
from nuronet2.activations import get_activation
from nuronet2.backend import N


def normalize_tuple(value, n, name):
    """Transforms a single int or iterable of ints into an int tuple.
    # Arguments
        value: The value to validate and convert. Could an int, or any iterable
          of ints.
        n: The size of the tuple to be returned.
        name: The name of the argument being validated, e.g. "strides" or
          "kernel_size". This is only used to format error messages.
    # Returns
        A tuple of n integers.
    # Raises
        ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
    return value_tuple


def normalize_padding(value):
    padding = value.lower()
    allowed = {'valid', 'same', 'causal'}
    if N.backend() == 'theano':
        allowed.add('full')
    if padding not in allowed:
        raise ValueError('The `padding` argument must be one of "valid", "same" (or "causal" for Conv1D). '
                         'Received: ' + str(padding))
    return padding


def convert_kernel(kernel):
    """Converts a Numpy kernel matrix from Theano format to TensorFlow format.
    Also works reciprocally, since the transformation is its own inverse.
    # Arguments
        kernel: Numpy array (3D, 4D or 5D).
    # Returns
        The converted kernel.
    # Raises
        ValueError: in case of invalid kernel shape or invalid data_format.
    """
    kernel = numpy.asarray(kernel)
    if not 3 <= kernel.ndim <= 5:
        raise ValueError('Invalid kernel shape:', kernel.shape)
    slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
    no_flip = (slice(None, None), slice(None, None))
    slices[-2:] = no_flip
    return numpy.copy(kernel[slices])


def conv_output_length(input_length, filter_size,
                       padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.
        dilation: dilation rate, integer.
    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def conv_input_length(output_length, filter_size, padding, stride):
    """Determines input length of a convolution given output length.
    # Arguments
        output_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full".
        stride: integer.
    # Returns
        The input length (integer).
    """
    if output_length is None:
        return None
    assert padding in {'same', 'valid', 'full'}
    if padding == 'same':
        pad = filter_size // 2
    elif padding == 'valid':
        pad = 0
    elif padding == 'full':
        pad = filter_size - 1
    return (output_length - 1) * stride - 2 * pad + filter_size


def deconv_length(dim_size, stride_size, kernel_size, padding):
    if dim_size is None:
        return None
    if padding == 'valid':
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == 'full':
        dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
    elif padding == 'same':
        dim_size = dim_size * stride_size
    return dim_size
    


class Conv(Layer):
    def __init__(self, rank,
                 filters, kernel_size,
                 strides=1, padding='valid',
                 dilation_rate=1,
                 kernel_factory="xavier_uniform",
                 bias_factory='zeros',
                 activation="linear",
                 kernel_regulariser=None,
                 bias_regulariser=None, **kwargs):
        self.rank = rank
        self.filters = filters
        self.kernel_size = normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = normalize_tuple(strides, rank, 'strides')
        self.padding = normalize_padding(padding)
        self.dilation_rate = normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = get_activation(activation)
        self.kernel_factory = get_weightfactory(kernel_factory)
        self.kernel_regulariser = get_regulariser(kernel_regulariser)
        self.bias_factory = get_weightfactory(bias_factory)
        self.bias_regulariser = get_regulariser(bias_regulariser)
        super(Conv, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        channel_axis = 1
        if(input_shape[channel_axis] is None):
            raise ValueError('The channel dimension of inputs should be defined')
        input_dim = input_shape[channel_axis]
        
        kernel_shape = (self.filters, input_dim) + self.kernel_size
        
        self.kernel = self.kernel_factory(shape=kernel_shape)
        self.bias = self.bias_factory(shape=(self.filters, ))
        self.trainable_weights = [self.kernel, self.bias]
        self._is_built = True
        
    def add_bias(self, x, bias):
        if(N.ndim(x) == 3):
            x += N.reshape(bias, (1, self.filters, 1))
        elif(N.ndim(x) == 4):
            x += N.reshape(bias, (1, self.filters, 1, 1))
        # RESERVED FOR 3D conv
        # elif(N.ndim(x) == 5)
        return x
    
    def prop_up(self, x):
        if(self.rank == 1):
            outputs = N.conv1d(x,
                               self.kernel,
                               strides=self.strides[0],
                               padding=self.padding,
                               dilation_rate=self.dilation_rate[0])
        if(self.rank == 2):
            outputs = N.conv2d(x,
                               self.kernel,
                               strides=self.strides,
                               padding=self.padding,
                               dilation_rate=self.dilation_rate)
        # Reserved for 3d convolutions
        """if(self.rank == 3)"""
        outputs = self.add_bias(outputs, self.bias)
        return self.activation(outputs)
    

        
    def get_cost(self):
        w_cost =  self.kernel_regulariser(self.kernel) if self.kernel_regulariser else N.cast(0.)
        b_cost = self.bias_regulariser(self.bias) if self.bias_regulariser else N.cast(0.)
        return w_cost + b_cost
        
    def get_output_shape(self, input_shape):
        space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(space[i],
                                         self.kernel_size[i],
                                         padding=self.padding,
                                         stride=self.strides[i],
                                         dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0], self.filters) + tuple(new_space)
        
        

class Conv2dLayer(Conv):
    def __init__(self, n, strides=(1, 1),
                 padding='valid', dilation_rate=(1, 1),
                 kernel_factory="xavier_uniform",
                 bias_factory='zeros',
                 activation="linear",
                 kernel_regulariser=None,
                 bias_regulariser=None, **kwargs):
        assert(isinstance(n, tuple) and len(n) == 3)
        filters = n[0]
        kernel_size = n[1:]
        super(Conv2dLayer, self).__init__(rank=2, filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides, padding=padding,
                                    dilation_rate=dilation_rate, activation=activation,
                                    kernel_factory=kernel_factory, bias_factory=bias_factory,
                                    kernel_regulariser=kernel_regulariser,
                                    bias_regulariser=bias_regulariser, **kwargs)
                                    
                                    
class Conv1dLayer(Conv):
    def __init__(self, n,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation='relu',
                 kernel_factory="xavier_uniform",
                 kernel_regulariser=None,                 
                 bias_factory='zeros',
                 bias_regulariser=None, **kwargs):
        
        filters = n[0]
        kernel_size = n[1]        
        super(Conv1dLayer, self).__init__(rank=1,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=strides,
                                          padding=padding,
                                          dilation_rate=dilation_rate,
                                          activation=activation,
                                          kernel_factory=kernel_factory,
                                          kernel_regulariser=kernel_regulariser,
                                          bias_factory=bias_factory,
                                          bias_regulariser=bias_regulariser,
                                          **kwargs)
        
        
        
        
        
        
        
        
        
        
        
        