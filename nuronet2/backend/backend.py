# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:02:54 2016

@author: Evander
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:32:03 2016

@author: Evander
"""

import numpy
from collections import defaultdict

class Backend(object):
    """
    Abstract class representing the interface between nuronet's software
    and the external 'hardware' it runs on.
    
    Future TODOS:
    Backend for Theano
    for Tensorflow
    for numpy
    for FPGAs
    """
    def __init__(self, rng_seed=None, default_dtype=numpy.float32):
        if(rng_seed is None):
            rng_seed = numpy.random.randint(1, 10e6)
        self._default_dtype = default_dtype
        self._rng_seed = rng_seed
        self.rng = self.createRNG(rng_seed)
        self._epsilon = 1e-7
        self.floatx = 'float32'
        self.uid_object = defaultdict(int)
        
    def get_uid(self, prefix=''):
        self.uid_object[prefix] += 1
        return self.uid_object[prefix]
        
    def reset_uids(self):
        self.uid_object = defaultdict(int)
        
    def floatx(self):
        return self.floatx
        
    @property
    def default_dtype(self):
        return self._default_dtype
        
    def int_shape(self, x):
        return x._nuro_shape
        
    def createRNG(self, seed=None):
        raise NotImplementedError()
        
    def shared(self, value, dtype=None, name=None):
        raise NotImplementedError()
        
    def variable(self, ndim=None, dtype=None, name=None):
        raise NotImplementedError()
        
    def scalar(self, dtype=None, name=None):
        raise NotImplementedError()
        
    def vector(self, dtype=None, name=None):
        raise NotImplementedError()
        
    def matrix(self, dtype=None, name=None):
        raise NotImplementedError()
        
    def tensor3(self, dtype=None, name=None):
        raise NotImplementedError()
        
    def tensor4(self, dtype=None, name=None):
        raise NotImplementedError()
        
    def shape(self, x):
        raise NotImplementedError()
        
    def ndim(self, x):
        raise NotImplementedError()
        
    def dtype(self, x):
        raise NotImplementedError()
        
    def zeros(self, shape, dtype=None, name=None):
        raise NotImplementedError()
        
    def ones(self, shape, dtype=None, name=None):
        raise NotImplementedError()
        
    def ones_like(self, x, dtype=None):
        raise NotImplementedError()
        
    def zeros_like(self, x):
        raise NotImplementedError()
        
    def count(self, x):
        raise NotImplementedError()
        
    def cast(self, x):
        raise NotImplementedError()
        
    def permute_dimensions(self, x, pattern):
        raise NotImplementedError()
        
    # LINEAR ALGEBRA OPS
        
    def dot(self, x, y):
        raise NotImplementedError()
        
    def batchedDot(self, x, y):
        #see both keras and neon implementatiosn
        raise NotImplementedError()
        
    def transpose(self, x):
        raise NotImplementedError()
        
    def gather(self, x, indices):
        raise NotImplementedError()
        
    #ELEMWISE OPS
        
    def max(self, x, axis=None, keepdims=False):
        raise NotImplementedError()
        
    def min(self, x, axis=None, keepdims=False):
        raise NotImplementedError()
        
    def sum(self, x, axis=None, keepdims=False):
        raise NotImplementedError()
        
    def prod(self, x, axis=None, keepdims=False):
        raise NotImplementedError()
        
    def mean(self, x, axis=None, keepdims=False):
        raise NotImplementedError()
        
    def std(self, x, axis=None, keepdims=False):
        raise NotImplementedError()
        
    def any(self, x, axis=None, keepdims=False):
        raise NotImplementedError()
        
    def argmax(self, x, axis=-1):
        raise NotImplementedError()
        
    def argmin(self, x, axis=-1):
        raise NotImplementedError()
        
    def square(self, x):
        raise NotImplementedError()
        
    def abs(self, x):
        raise NotImplementedError()
        
    def sqrt(self, x):
        raise NotImplementedError()
        
    def exp(self, x):
        raise NotImplementedError()
        
    def exp2(self, x):
        raise NotImplementedError()
        
    def log(self, x):
        raise NotImplementedError()
        
    def round(self, x):
        raise NotImplementedError()
        
    def sign(self, x):
        raise NotImplementedError()
        
    def pow(self, x, a):
        raise NotImplementedError()
        
    def clip(self, x, min_value, max_value):
        raise NotImplementedError()

    def equal(self, x, y):
        raise NotImplementedError()
        
    def neq(self, x, y):
        raise NotImplementedError()
        
    def maximum(self, x, y):
        raise NotImplementedError()
        
    def minimum(self, x, y):
        raise NotImplementedError()
        
    def sin(self, x):
        raise NotImplementedError()
        
    def cos(self, x):
        raise NotImplementedError()
        
        
    #RESHAPING OPS
        
    def concat(self, tensors, axis=-1):
        raise NotImplementedError()
        
    def reshape(self, x, shape):
        raise NotImplementedError()
        
    def dimshuffle(self, x, pattern):
        raise NotImplementedError()
        
    def tile(self, x, n):
        raise NotImplementedError()
        
    def flatten(self, x, **kwargs):
        raise NotImplementedError()
        
    def batch_flatten(self, x, **kwargs):
        raise NotImplementedError()
        
    #VALUE OPS
        
    def get_value(self, x):
        raise NotImplementedError()
        
    def set_value(self, x, value):
        raise NotImplementedError()
        
    ##GRAPH MANIPULATION
    def function(self, inputs, outputs, updates=[], **kwargs):
        raise NotImplementedError()
        
    def gradients(self, loss, variables):
        raise NotImplementedError()
        
    #CONTROL FLOW OPS
    def lt(self, a, b):
        raise NotImplementedError()
        
    def le(self, a, b):
        raise NotImplementedError()
        
    def gt(self, a, b):
        raise NotImplementedError()
        
    def ge(self, a, b):
        raise NotImplementedError()
        
    def switch(self, if_condition, then_expression, else_expression):
        raise NotImplementedError()
        
    # RANDOM GENERATORS
    def rng_normal(self, shape, mean=0., std=1, dtype=None):
        raise NotImplementedError()
        
    def rng_uniform(self, shape, low=0., high=1., dtype=None):
        raise NotImplementedError()
        
    def rng_binomial(self, shape, p=0., dtype=None):
        raise NotImplementedError()
        
    #NeuralNet OPS
    def conv2d(self, x, kernel, strides=(1, 1), border_mode='valid',
               image_shape=None, filter_shape=None):
        raise NotImplementedError()
        
    def pool2d(self, x, pool_size, strides=(1, 1), border_mode='valid',
               pool_mode='max'):
        raise NotImplementedError()
        
    def recurrence(self, step_function, inputs, initial_states,
            unroll=False,
            go_backwards=False, constants=None, input_length=None):
        raise NotImplementedError()

    def l2_norm(self, x, axis):
        raise NotImplementedError()
        
    def dropout(self, x, level):
        raise NotImplementedError()
        
    def tanh(self, x):
        raise NotImplementedError()
        
    def hard_sigmoid(self, x):
        raise NotImplementedError()
        
    def sigmoid(self, x):
        raise NotImplementedError()
        
    def binary_crossentropy(self, output, target, logit=False):
        raise NotImplementedError()
        
    def categorical_crossentropy(self, output, target, logit=False):
        raise NotImplementedError()
        
    def softmax(self, x):
        raise NotImplementedError()
        
    def softplus(self, x):
        raise NotImplementedError()
        
    def relu(self, x, alpha, max_value=None):
        raise NotImplementedError()
        
        
        
        
        

        
    
        
