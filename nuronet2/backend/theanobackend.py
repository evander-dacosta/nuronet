# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 21:45:13 2016

@author: Evander
"""
import numpy
import inspect
import theano
import theano.tensor as T
import theano.tensor.signal.pool as pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from backend import Backend

class TheanoBackend(Backend):
    def __init__(self, rng_seed=None, default_dtype=theano.config.floatX):
        Backend.__init__(self, rng_seed=rng_seed, default_dtype=default_dtype)
        
    def backend(self):
        return 'theano'
        
    def createRNG(self, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        return RandomStreams(seed=seed)
        
    def shared(self, value, dtype=None, name=None):
        if(dtype is None):
            dtype = self._default_dtype
        value = numpy.asarray(value, dtype=dtype)
        return theano.shared(value=value, name=name, strict=False)
        
    def variable(self, ndim, dtype=None, name=None):
        if(dtype is None):
            dtype = self._default_dtype
        broadcast = (False, ) * ndim
        ret = T.TensorType(dtype=dtype, broadcastable=broadcast)(name)
        ret._nuro_shape = tuple([None for _ in range(ndim)])
        ret._nuro_history = None
        return ret

    def constant(self, value, dtype=None, shape=None, name=None):
        if(dtype is None):
            dtype = self.floatx
        if(shape is None):
            shape = ()
        np_value = value * numpy.ones(shape)
        const = T.constant(np_value, dtype=dtype, name=name)
        const._nuro_shape = shape
        return const

    def scalar(self, dtype=None, name=None):
        return self.variable(ndim=0, dtype=dtype, name=name)
        
    def vector(self, dtype=None, name=None):
        return self.variable(ndim=1, dtype=dtype, name=name)
        
    def matrix(self, dtype=None, name=None):
        return self.variable(ndim=2, dtype=dtype, name=name)
        
    def tensor3(self, dtype=None, name=None):
        return self.variable(ndim=3, dtype=dtype, name=name)
        
    def tensor4(self, dtype=None, name=None):
        return self.variable(ndim=4, dtype=dtype, name=name)
        
    def shared_shape(self, x):
        return x.get_value(borrow=True, return_internal_type=True).shape
        
    def shape(self, x):
        return x.shape
        
    def int_shape(self, x):
        """
        Returns the shape of a tensor as a tuple of ints
        or None entries. Works only with Tensorflow
        """
        if hasattr(x, '_nuro_shape'):
            return x._nuro_shape
        else:
            raise Exception('Not a Nuronet tensor:', x)
        
    def ndim(self, x):
        return x.ndim
        
    def dtype(self, x):
        return x.dtype
    
    def zeros(self, shape, dtype=None, name=None):
        if(dtype is None):
            dtype = self._default_dtype
        return self.shared(numpy.zeros(shape), dtype, name)
        
    def ones(self, shape, dtype=None, name=None):
        if(dtype is None):
            dtype = self._default_dtype
        return self.shared(numpy.ones(shape), dtype, name)        
        
    def ones_like(self, x, dtype=None):
        if(dtype is None):
            dtype = self.dtype(x)
        return T.ones_like(x, dtype=dtype)

    def zeros_like(self, x, dtype=None):
        if(dtype is None):
            dtype = self.dtype(x)
        return T.zeros_like(x, dtype=dtype)
        
    def count(self, x):
        return numpy.prod(x.shape.eval())
        
    def cast(self, x, dtype=None):
        if(dtype is None):
            dtype = self._default_dtype
        return T.cast(x, dtype)
        
    def stack(self, x):
        return T.stack(*x)
        
    def repeat(self, x, n):
        """Repeat a 2D tensor.
        If x has shape (samples, dim) and n=2,
        the output will have shape (samples, 2, dim).
        """
        assert x.ndim == 2
        x = x.dimshuffle((0, 'x', 1))
        return T.extra_ops.repeat(x, n, axis=1)
        
    def permute_dimensions(self, x, pattern):
        pattern = tuple(pattern)
        y = x.dimshuffle(pattern)
        if(hasattr(x, '_nuro_shape')):
            y._nuro_shape = tuple(numpy.array(x._nuro_shape)[list(pattern)])
        return y

    # LINEAR ALGEBRA OPS
        
    def dot(self, x, y):
        return T.dot(x, y)
        
    def batchedDot(self, x, y):
        #see both keras and neon implementatiosn
        raise NotImplementedError()
        
    def transpose(self, x):
        return T.transpose(x)
        
    def gather(self, x, indices):
        return x[indices]
        
    def batch_normalization(self, x, mean, var, beta, gamma, epsilon=1e-3):
        """Apply batch normalization on x given mean, var, beta and gamma.
        """
        if gamma is None:
            gamma = self.ones_like(var)
        if beta is None:
            beta = self.zeros_like(mean)
    
        if mean.ndim == 1:
            # based on TensorFlow's default: normalize along rightmost dimension
            reduction_axes = list(range(x.ndim - 1))
        else:
            reduction_axes = [i for i in range(x.ndim) if mean.broadcastable[i]]
    
        return T.nnet.bn.batch_normalization_test(
            x, gamma, beta, mean, var, reduction_axes, epsilon)
            
    def normalize_batch_in_training(self, x, gamma, beta,
                                    reduction_axes, epsilon=1e-3):
        """Computes mean and std for batch then apply batch_normalization on batch.
        """
        if gamma is None:
            if beta is None:
                gamma = self.ones_like(x)
            else:
                gamma = self.ones_like(beta)
        if beta is None:
            if gamma is None:
                beta = self.zeros_like(x)
            beta = self.zeros_like(gamma)
    
        normed, mean, stdinv = T.nnet.bn.batch_normalization_train(
            x, gamma, beta, reduction_axes, epsilon)
    
        return normed, mean, T.inv(stdinv ** 2)
        
    #ELEMWISE OPS
        
    def max(self, x, axis=None, keepdims=False):
        return T.max(x, axis=axis, keepdims=keepdims)
        
    def min(self, x, axis=None, keepdims=False):
        return T.min(x, axis=axis, keepdims=keepdims)
        
    def sum(self, x, axis=None, keepdims=False):
        return T.sum(x, axis=axis, keepdims=keepdims)
        
    def prod(self, x, axis=None, keepdims=False):
        return T.prod(x, axis=axis, keepdims=keepdims)
        
    def mean(self, x, axis=None, keepdims=False):
        return T.mean(x, axis=axis, keepdims=keepdims)
        
    def var(self, x, axis=None, keepdims=False):
        return T.var(x, axis=axis, keepdims=keepdims)
        
    def std(self, x, axis=None, keepdims=False):
        return T.std(x, axis=axis, keepdims=keepdims)
        
    def any(self, x, axis=None, keepdims=False):
        return T.any(x, axis=axis, keepdims=keepdims)
        
    def argmax(self, x, axis=-1):
        return T.argmax(x, axis=axis, keepdims=False)
        
    def argmin(self, x, axis=-1):
        return T.argmin(x, axis=axis, keepdims=False)
        
    def square(self, x):
        return T.sqr(x)
        
    def abs(self, x):
        return T.abs_(x)
        
    def sqrt(self, x):
        x = self.clip(x, 0., numpy.inf)
        return T.sqrt(x)
        
    def exp(self, x):
        return T.exp(x)
        
    def exp2(self, x):
        return T.exp2(x)
        
    def log(self, x):
        return T.log(x)
        
    def round(self, x):
        return T.round(x)
        
    def sign(self, x):
        return T.sgn(x)
        
    def pow(self, x, a):
        return T.pow(x, a)
        
    def clip(self, x, min_value, max_value):
        if(max_value < min_value):
            max_value = min_value
        return T.clip(x, min_value, max_value)

    def equal(self, x, y):
        return T.eq(x, y)
        
    def neq(self, x, y):
        return T.neq(x, y)
        
    def maximum(self, x, y):
        return T.maximum(x, y)
        
    def minimum(self, x, y):
        return T.minimum(x, y)
    
    def sin(self, x):
        return T.sin(x)
        
    def cos(self, x):
        return T.cos(x)
        
    #RESHAPING OPS
        
    def concat(self, tensors, axis=-1):
        return T.concatenate(tensors, axis=axis)
        
    def reshape(self, x, shape):
        return T.reshape(x, shape)
        
    def expand_dim(self, x, dim=-1):
        pattern = [i for i in range(x.type.ndim)]
        if(dim < 0):
            if x.type.ndim == 0:
                dim = 0
            else:
                dim = dim % x.type.ndim + 1
        pattern.insert(dim, 'x')
        return x.dimshuffle(pattern)
        
    def dimshuffle(self, x, pattern):
        pattern = tuple(pattern)
        return x.dimshuffle(pattern)
        
    def tile(self, x, n):
        return T.tile(x, n)
        
    def flatten(self, x, **kwargs):
        return T.flatten(x, **kwargs)
        
    def batch_flatten(self, x, **kwargs):
        x = T.reshape(x, (x.shape[0], T.prod(x.shape) // x.shape[0]))
        return x
        
    #VALUE OPS
        
    def get_value(self, x):
        if(not hasattr(x, 'get_value')):
            raise Exception("Cannot call get_value() on a theano variable" + 
            "that isn't a shared type")
        return x.get_value()
        
    def set_value(self, x, value):
        if(not hasattr(x, 'set_value')):
            raise Exception("Cannot call set_value() on a variable" + 
            "that isn't a shared type")
        return x.set_value(numpy.asarray(value, dtype=x.dtype))
        
    ##GRAPH MANIPULATION
    def function(self, inputs, outputs, updates=[], **kwargs):
        if(len(kwargs) > 0):
            fArgs = inspect.getargspec(theano.function)[0]
            for key in kwargs.keys():
                if(key not in fArgs):
                    raise ValueError("Invalid argument {} passed to " +
                    "function(**kwargs)".format(key))
        return theano.function(inputs, outputs, updates=updates,
                               allow_input_downcast=True,
                               on_unused_input='warn',
                               **kwargs)
                               
    def gradients(self, loss, variables):
        return T.grad(loss, variables)
        
    #CONTROL FLOW OPS
        
    def lt(self, a, b):
        return T.lt(a, b)
        
    def le(self, a, b):
        return T.le(a, b)
        
    def gt(self, a, b):
        return T.gt(a, b)
        
    def ge(self, a, b):
        return T.ge(a, b)
        
    def switch(self, if_condition, then_expression, else_expression):
        if(isinstance(if_condition, bool)):
            if_condition = 1 if if_condition else 0
        if callable(then_expression):
            then_expression = then_expression()
        if callable(else_expression):
            else_expression = else_expression()
        return T.switch(if_condition, then_expression, else_expression)
        
    # RANDOM GENERATORS
    def rng_normal(self, shape, mean=0., std=1, dtype=None, seed=None):
        if(dtype is None):
            dtype = self._default_dtype
        return self.createRNG(seed=seed).normal(size=shape, avg=mean, 
                                                std=std, dtype=dtype)
        
    def rng_uniform(self, shape, low=0., high=1., dtype=None, seed=None):
        if(dtype is None):
            dtype = self._default_dtype
        return self.createRNG(seed=seed).uniform(shape, low=low, high=high,
                                                    dtype=dtype)
        
    def rng_binomial(self, shape, p=0., dtype=None, seed=None):
        if(dtype is None):
            dtype = self._default_dtype
        return self.createRNG(seed=seed).binomial(shape, p=p, dtype=dtype)
        
    def rng_truncated_normal(self, shape, mean=0., std=1., dtype=None, seed=None):
        if(dtype is None):
            dtype = self._default_dtype
        rng = self.createRNG(seed=seed)
        normal_tensor = rng.normal(size=shape, avg=mean, std=std, dtype=dtype)
        return self.clip(normal_tensor, mean - 2*std, mean + 2 * std)
        
        
    #NEURAL NETWORK OPS
    def conv2d(self, x, kernel, strides=(1, 1), border_mode='valid',
               image_shape=None, filter_shape=None):
        if(border_mode not in ('valid', 'full')):
            raise Exception("Border mode not supported:{}".format(border_mode))
        
        def int_else_none(value):
            try:
                return int(value)
            except TypeError:
                return None
                
        if(image_shape is not None):
            image_shape = tuple(int_else_none(v) for v in image_shape)
            
        if(filter_shape is not None):
            filter_shape = tuple(int_else_none(v) for v in filter_shape)
            
        out = T.nnet.conv2d(x, kernel, border_mode=border_mode,
                            subsample=strides,
                            input_shape=image_shape,
                            filter_shape=filter_shape)
        return out
        
        
    def pool2d(self, x, pool_size, strides=(1, 1), padding=(0, 0),
               ignore_border=False,
               pool_mode='max'):
        if(pool_mode not in ('max', 'avg')):
            raise Exception("Unknown pool_mode" + \
            ":{}. Required 'avg' or 'max'".format(pool_mode))
        if(pool_mode == 'max'):
            poolOut = pool.pool_2d(x, ds=pool_size, st=strides,
                                   ignore_border=ignore_border,
                                   padding=padding,
                                    mode='max')
        else:
            poolOut = pool.pool_2d(x, ds=pool_size, st=strides,
                                   ignore_border=ignore_border,
                                   padding=padding, 
                                   mode='average_exc_pad')
                                   
        return poolOut

    def recurrence(self, step_function, inputs, initial_states,
            unroll=False,
            go_backwards=False, constants=None,
            input_length=None):
        ndim = inputs.ndim
        if(ndim < 3):
            raise Exception("Inputs should have ndim > 3. Given {}".format(ndim))
        
        if(unroll and input_length is None):
            raise Exception("When unroll is specified, input_length must"+\
                            " also be given")
        axes = [1, 0] + list(range(2, ndim))
        inputs = inputs.dimshuffle(axes)
        
        if(constants is None):
            constants = []
            
        if(unroll):
            indices = list(range(input_length))
            if(go_backwards):
                indices = indices[::-1]
            next_outputs = []
            next_states = []
            currentStates = initial_states
            for i in indices:
                output, currentStates = step_function(inputs[i],
                                                      currentStates + constants)
                next_outputs.append(output)
                next_states.append(currentStates)
            outputs = T.stack(*next_outputs)
            currentStates = []
            for i in range(len(next_states[-1])):
                currentStates.append(T.stack(*[state[i] for state in next_states]))
            
        else:
            def _stepFunc(input, *states):
                output, newStates = step_function(input, states)
                return [output] + newStates
                
            if(len(initial_states) > 0):
                initial_states[0] = T.unbroadcast(initial_states[0], 1)
                
            results, _ = theano.scan(
                _stepFunc,
                sequences=inputs,
                outputs_info=[None] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)
                
            if(type(results) is list):
                outputs=results[0]
                currentStates = results[1:]
            else:
                outputs = results
                currentStates = []
                
        outputs = T.squeeze(outputs)
        last_output = outputs[-1]
        
        axes = [1, 0] + list(range(2, outputs.ndim))
        outputs = outputs.dimshuffle(axes)
        currentStates = T.stack(*[T.squeeze(state[-1]) for state in currentStates])
        return last_output, outputs, currentStates
        
        

    def l2_norm(self, x, axis):
        norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
        return x / norm
        
    def dropout(self, x, probability):
        if(probability < 0 or probability >= 1):
            raise Exception("probability must be in range [0, 1] not {}" \
                            .format(probability))
        retain = self.cast(1. - probability)
        x *= self.rng_binomial(x.shape, p=retain, dtype=x.dtype)
        x /= retain
        return x
        
    def tanh(self, x):
        return T.tanh(x)
        
    def hard_sigmoid(self, x):
        return T.nnet.hard_sigmoid(x)
        
    def sigmoid(self, x):
        return T.nnet.sigmoid(x)
        
    def binary_crossentropy(self, output, target, logit=False):
        if(logit):
            output = T.nnet.sigmoid(output)
        output = T.clip(output, self._epsilon, 1. - self._epsilon)
        return T.nnet.binary_crossentropy(output, target)
        
    def categorical_crossentropy(self, output, target, logit=False):
        if(logit):
            output = T.nnet.softmax(output)
        else:
            #scale it so that the probabilities sum to 1
            output /= output.sum(axis=-1, keepdims=True)
        #avoid numerical instability by clipping probabilities
        output = T.clip(output, self._epsilon, 1.0 - self._epsilon)
        return T.nnet.categorical_crossentropy(output, target)
        
        
    def softmax(self, x):
        return T.nnet.softmax(x)
        
    def softplus(self, x):
        return T.nnet.softplus(x)
        
    def relu(self, x, alpha, max_value=None):
        x = T.nnet.relu(x, alpha)
        if max_value is not None:
            x = T.minimum(x, max_value)
        return x
        
        
        
        
        
        