# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:13:19 2016

@author: evander
"""

import numpy
import inspect
import os
import warnings
import copy
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from backend import Backend

default_dtype = tf.float32

def _str_dtype(dtype):
    if dtype == None:
        global default_dtype
        return default_dtype
    if dtype == 'float16':
        return tf.float16
    if dtype == 'float32':
        return tf.float32
    elif dtype == 'float64':
        return tf.float64
    elif dtype == 'int16':
        return tf.int16
    elif dtype == 'int32':
        return tf.int32
    elif dtype == 'int64':
        return tf.int64
    elif dtype == 'int8':
        return tf.int8
    elif dtype == 'uint8':
        return tf.int8
    elif dtype == 'uint16':
        return tf.uint16
    else:
        raise ValueError('Unsupported dtype:', dtype)
        
class Function(object):
    def __init__(self, backend, inputs, outputs, updates=[], sole_output=False):
        assert(isinstance(inputs, (list, tuple)))
        assert(isinstance(outputs, (list, tuple)))
        assert(isinstance(updates, (list, tuple)))
        self.backend = backend        
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.sole_output = sole_output
        with tf.control_dependencies(self.outputs):
            updates_ops = []
            for update in updates:
                if(type(update is tuple)):
                    val, new_val = update
                    updates_ops.append(tf.assign(val, new_val))
                else:
                    updates_ops.append(update)
            self.updates_op = tf.group(*updates_ops)
            
    def __call__(self, *args):
        inputs = args
        assert(isinstance(inputs, (list, tuple)))
        names = [getattr(v, 'name', None) for v in self.inputs]
        feed_dict = dict(zip(names, inputs))
        session = self.backend.get_session()
        updated = session.run(self.outputs + [self.updates_op], feed_dict=feed_dict)
        if(self.sole_output):
            return updated[:len(self.outputs)][0]
        return updated[:len(self.outputs)]
        

class TensorflowBackend(Backend):
    def __init__(self, rng_seed=None):
        global default_dtype
        Backend.__init__(self, rng_seed=rng_seed, 
                         default_dtype=default_dtype.name)
        self.manual_var_init = False
        self._session = None
        
        
    #--------------------Tensorflow-related stuff-------------------------#
    def _initialize_variables(self):
        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if(not hasattr(v, '_n_initialized') or not v._n_initialized):
                uninitialized_variables.append(v)
                v._n_initialized = True
        if uninitialized_variables:
            sess = self.get_session()
            sess.run(tf.variables_initializer(uninitialized_variables))
            
    def eval(self, variables):
        f = self.function([], variables)
        return f()

    def backend(self):
        return 'tensorflow'

    @property
    def session(self):
        return self.get_session()

    @property
    def manual_var(self):
        return self.manual_var_init
        
    @manual_var.setter
    def manual_var(self, value):
        assert(isinstance(value, bool))
        self.manual_var_init = value
        
    def clear_session(self):
        tf.reset_default_graph()
        self._session = None
        
    def get_session(self):
        """
        Returns the TF session to be used by the backend
        If a default TF session is available, that will be used.
        Otherwise, it will return the global Nuronet session.
        Sessions can be manually set using N.set_session(session).
        """
        
        if(tf.get_default_session() is not None):
            return tf.get_default_session()
        if(self._session is None):
            if(not os.environ.get("OMP_NUM_THREADS")):
                self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            else:
                n_threads = int(os.environ.get("OMP_NUM_THREADS"))
                self._session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=n_threads,
                                                           allow_soft_placement=True))
        with self._session.graph.as_default():
            self._initialize_variables()
        return self._session
        
    def set_session(self, session):
        self._session = session
        
        
        
    def to_tensor(self, x, dtype):
        x = tf.convert_to_tensor(x)
        if(x.dtype != dtype):
            x = tf.cast(x, dtype)
        return x
    #---------------------------------------------------------------------#
            
        
    def createRNG(self, seed=None):
        if(seed is None):
            self._rng_seed = numpy.random.randint(1e7)
        else:
            self._rng_seed = seed
            
    def shared(self, value, dtype=None, name=None):
        v = tf.Variable(value, dtype=_str_dtype(dtype), name=name)
        if(self.manual_var):
            return v
        if(tf.get_default_graph() is self.session.graph):
            try:
                self.get_session().run(v.initializer)
            except tf.errors.InvalidArgumentError:
                warnings.warn("Could not automatically initialize variable," + \
                              "make sure to do it manually, e.g. via tf.initialize_all_variables()")
        else:
            warnings.warn('The default Tensorflow graph is not the graph '
                    'associated with the Tensorflow session currently registered '
                    'with Nuronet. Nuronet was therefore not able to automatically '
                    'initialize a variable. Consider setting the proper session '
                    'to Nuronet with N.set_session(session)')
        return v        
        
    def variable(self, ndim, dtype=None, name=None):
        shape = tuple([None for _ in range(ndim)])
        x = tf.placeholder(dtype=_str_dtype(dtype), shape=shape, name=name)
        x._nuro_shape = shape
        x._nuro_history = None
        return x
        
    def constant(self, value, dtype=None, shape=None, name=None):
        if(dtype is None):
            dtype = self.floatx
        return tf.constant(value, dtype=dtype, shape=shape, name=name)
        
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
        shape = x.get_shape()
        return tuple([i.__int__() for i in shape])

    def shape(self, x):
        return tf.shape(x)
        
    def int_shape(self, x):
        """
        Returns the shape of a tensor as a tuple of ints
        or None entries. Works only with Tensorflow
        """
        if(hasattr(x, '_nuro_shape')):
            return x._nuro_shape
        shape = x.get_shape()
        try:
            return tuple([i.__int__() for i in shape])
        except ValueError:
            return None
        
    def ndim(self, x):
        dims = x.get_shape()._dims
        if(dims is not None):
            return len(dims)
        return None
        
    def dtype(self, x):
        return x.dtype.name
        
    def zeros(self, shape, dtype=None, name=None):
        shape = tuple(map(int, shape))
        tf_dtype = _str_dtype(dtype)
        return self.shared(tf.constant_initializer(0., dtype=tf_dtype)(shape), 
                           dtype, name)
        
    def ones(self, shape, dtype=None, name=None):
        shape = tuple(map(int, shape))
        tf_dtype = _str_dtype(dtype)
        return self.shared(tf.constant_initializer(1., dtype=tf_dtype)(shape), 
                           dtype, name)
        
    def ones_like(self, x, name=None):
        return tf.ones_like(x, name=name)
        
    def zeros_like(self, x, name=None):
        return tf.zeros_like(x, name=name)
        
    def count(self, x):
        shape = x.get_shape()
        return numpy.prod([shape[i]._value for i in range(len(shape))])
        
    def cast(self, x, dtype=None):
        return tf.cast(x, dtype=_str_dtype(dtype))
        
    def stack(self, x):
        return tf.stack(x)

        
    def repeat(self, x, n):
        """Repeats a 2D tensor.
        if x has shape (samples, dim) and n is 2,
        the output will have shape (samples, 2, dim).
        # Returns
            A tensor.
        """
        assert self.ndim(x) == 2
        x = tf.expand_dims(x, 1)
        pattern = self.stack([1, n, 1])
        return tf.tile(x, pattern)
        
    def permute_dimensions(self, x, pattern):
        return tf.transpose(x, perm=pattern)
        
    # LINEAR ALGEBRA OPS
        
    def dot(self, x, y):
        if(self.ndim(x) is not None and (self.ndim(x) > 2 or self.ndim(y) > 2)):
            x_shape = []
            for i, s in zip(self.int_shape(x), tf.unstack(tf.shape(x))):
                if(i is not None):
                    x_shape.append(i)
                else:
                    x_shape.append(s)
            x_shape = tuple(x_shape)
            y_shape = []
            for i, s in zip(self.int_shape(y), tf.unstack(tf.shape(y))):
                if(i is not None):
                    y_shape.append(i)
                else:
                    y_shape.append(s)
            y_shape = tuple(y_shape)
            y_permute_dim = list(range(self.ndim(y)))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = tf.reshape(x, [-1, x_shape[-1]])
            yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), 
                            [y_shape[-2], -1])
            return tf.reshape(tf.matmul(xt, yt),
                              x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
        out = tf.matmul(x, y)
        return out
        
    def batchedDot(self, x, y):
        #see both keras and neon implementatiosn
        raise NotImplementedError()
        
    def transpose(self, x):
        return tf.transpose(x)
        
    def gather(self, x, indices):
        return tf.gather(x, indices)
        
    def batch_normalization(self, x, mean, var, beta, gamma, epsilon=1e-3):
        """Applies batch normalization on x given mean, var, beta and gamma.
        I.e. returns:
        `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`
        # Arguments
            x: Input tensor or variable.
            mean: Mean of batch.
            var: Variance of batch.
            beta: Tensor with which to center the input.
            gamma: Tensor by which to scale the input.
            epsilon: Fuzz factor.
        # Returns
            A tensor.
        """
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
        
    #ELEMWISE OPS
        
    def _axis(self, axis, ndim):
        if(type(axis) is tuple):
            axis = list(axis)
        if(type(axis) is list):
            for i, a in enumerate(axis):
                if(a is not None and a < 0):
                    axis[i] = a % ndim
        else:
            if(axis is not None and axis < 0):
                axis = axis % ndim
        return axis
        
    def max(self, x, axis=None, keepdims=False):
        axis = self._axis(axis, self.ndim(x))
        return tf.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)
        
    def min(self, x, axis=None, keepdims=False):
        axis = self._axis(axis, self.ndim(x))
        return tf.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)
        
    def sum(self, x, axis=None, keepdims=False):
        axis = self._axis(axis, self.ndim(x))
        return tf.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)
        
    def prod(self, x, axis=None, keepdims=False):
        axis = self._axis(axis, self.ndim(x))
        return tf.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)
        
    def mean(self, x, axis=None, keepdims=False):
        global default_dtype
        axis = self._axis(axis, self.ndim(x))
        if(x.dtype.base_dtype == tf.bool):
            x = tf.cast(x, default_dtype)
        return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)
    
    def var(self, x, axis=None, keepdims=False):
        global default_dtype
        axis = self._axis(axis, self.ndim(x))
        if(x.dtype.base_dtype == tf.bool):
            x = tf.cast(x, default_dtype)
        mean = tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)
        squared = tf.square(x - mean)
        return tf.reduce_mean(squared, reduction_indices=axis, 
                              keep_dims=keepdims)
        
    def std(self, x, axis=None, keepdims=False):
        return self.sqrt(self.var(x, axis=axis, keepdims=keepdims))
        
    def any(self, x, axis=None, keepdims=False):
        axis = self._axis(axis, self.ndim(x))
        x = tf.cast(x, tf.bool)
        x = tf.reduce_any(x, reduction_axis=axis, keep_dims=keepdims)
        return tf.cast(x, tf.uint8)
        
    def all(self, x, axis=None, keepdims=False):
        axis = self._axis(axis, self.ndim(x))
        x = tf.cast(x, tf.bool)
        x = tf.reduce_all(x, reduction_axis=axis, keep_dims=keepdims)
        return tf.cast(x, tf.uint8)
        
    def argmax(self, x, axis=-1):
        if(axis < 0):
            axis = axis % len(x.get_shape())
        return tf.argmax(x, axis)
        
    def argmin(self, x, axis=-1):
        if(axis < 0):
            axis = axis % len(x.get_shape())
        return tf.argmin(x, axis)
        
    def square(self, x):
        return tf.square(x)
        
    def abs(self, x):
        return tf.abs(x)
        
    def sqrt(self, x):
        z = self.to_tensor(0., x.dtype.base_dtype)
        inf = self.to_tensor(numpy.inf, x.dtype.base_dtype)
        x = tf.clip_by_value(x, z, inf)
        return tf.sqrt(x)
        
    def exp(self, x):
        return tf.exp(x)
        
    def exp2(self, x):
        return tf.exp(x)
        
    def log(self, x):
        return tf.log(x)
        
    def round(self, x):
        return tf.round(x)
        
    def sign(self, x):
        return tf.sign(x)
        
    def pow(self, x, a):
        return tf.pow(x, a)
        
    def clip(self, x, min_value, max_value):
        if(max_value < min_value):
            max_value = min_value
        min_value = self.to_tensor(min_value, x.dtype.base_dtype)
        max_value = self.to_tensor(max_value, x.dtype.base_dtype)
        return tf.clip_by_value(x, min_value, max_value)

    def equal(self, x, y):
        return tf.equal(x, y)
        
    def neq(self, x, y):
        return tf.not_equal(x, y)
        
    def maximum(self, x, y):
        return tf.maximum(x, y)
        
    def minimum(self, x, y):
        return tf.minimum(x, y)
        
    def sin(self, x):
        return tf.sin(x)
        
    def cos(self, x):
        return tf.cos(x)
        
        
    #RESHAPING OPS
        
    def normalize_batch_in_training(self, x, gamma, beta,
                                    reduction_axes, epsilon=1e-3):
        """Computes mean and std for batch then apply batch_normalization on batch.
        # Arguments
            x: Input tensor or variable.
            gamma: Tensor by which to scale the input.
            beta: Tensor with which to center the input.
            reduction_axes: iterable of integers,
                axes over which to normalize.
            epsilon: Fuzz factor.
        # Returns
            A tuple length of 3, `(normalized_tensor, mean, variance)`.
        """
        mean, var = tf.nn.moments(x, reduction_axes,
                                  shift=None, name=None, keep_dims=False)
        if sorted(reduction_axes) == list(range(self.ndim(x)))[:-1]:
            normed = tf.nn.batch_normalization(x, mean, var,
                                               beta, gamma,
                                               epsilon)
        else:
            # need broadcasting
            target_shape = []
            for axis in range(self.ndim(x)):
                if axis in reduction_axes:
                    target_shape.append(1)
                else:
                    target_shape.append(tf.shape(x)[axis])
            target_shape = tf.stack(target_shape)
    
            broadcast_mean = tf.reshape(mean, target_shape)
            broadcast_var = tf.reshape(var, target_shape)
            if gamma is None:
                broadcast_gamma = None
            else:
                broadcast_gamma = tf.reshape(gamma, target_shape)
            if beta is None:
                broadcast_beta = None
            else:
                broadcast_beta = tf.reshape(beta, target_shape)
            normed = tf.nn.batch_normalization(x, broadcast_mean, broadcast_var,
                                               broadcast_beta, broadcast_gamma,
                                               epsilon)
        return normed, mean, var
            
    def concat(self, tensors, axis=-1):
        if(axis < 0):
            rank = self.ndim(tensors[0])
            if(rank):
                axis %= rank
            else:
                axis = 0
        return tf.concat(tensors, axis)
        
    def reshape(self, x, shape):
        return tf.reshape(x, shape)
        
    def expand_dim(self, x, dim=-1):
        return tf.expand_dims(x, dim)
        
    def squeeze(self, x, axis):
        return tf.squeeze(x, [axis])
        
    def dimshuffle(self, x, pattern):
        return tf.transpose(x, perm=pattern)
        
    def tile(self, x, n):
        if(not hasattr(n, 'shape') and not hasattr(n, '__len__') and not hasattr(n, '_shape')):
            n = [n]
        return tf.tile(x, n)
        
    def flatten(self, x, **kwargs):
        return tf.reshape(x, [-1])
        
    def batch_flatten(self, x, **kwargs):
        x = tf.reshape(x, tf.stack([-1, self.prod(self.shape(x)[1:])]))
        return x
        
    #VALUE OPS
        
    def get_value(self, x):
        return x.eval(session=self.session)
        
    def set_value(self, x, value):
        value = numpy.asarray(value)
        tf_dtype = _str_dtype(x.dtype.name.split('_')[0])
        if(hasattr(x, '_assign_placeholder')):
            assign_placeholder = x._assign_placeholder
            assign_op = x._assign_op
        else:
            assign_placeholder = tf.placeholder(tf_dtype, shape=value.shape)
            assign_op = x.assign(assign_placeholder)
            x._assign_placeholder = assign_placeholder
            x._assign_op = assign_op
        self.session.run(assign_op, feed_dict={assign_placeholder:value})
        
    ##GRAPH MANIPULATION
    def function(self, inputs, outputs, updates=[], **kwargs):
        sole_output = False
        if(not isinstance(outputs, (list, tuple))):
            outputs = [outputs]
            sole_output = True
        return Function(self, inputs, outputs, updates=updates, sole_output=sole_output)
        
    def gradients(self, loss, variables):
        return tf.gradients(loss, variables, colocate_gradients_with_ops=True)
        
    #CONTROL FLOW OPS
    def lt(self, a, b):
        return tf.less(a, b)
        
    def le(self, a, b):
        return tf.less_equal(a, b)
        
    def gt(self, a, b):
        return tf.greater(a, b)
        
    def ge(self, a, b):
        return tf.greater_equal(a, b)
        
    def switch(self, if_condition, then_expression, else_expression):
        if(if_condition.dtype != tf.bool):
            if_condition = tf.cast(if_condition, 'bool')
        if(not callable(then_expression)):
            def then_expression_fn():
                return then_expression
        else:
            then_expression_fn = then_expression
        if(not callable(else_expression)):
            def else_expression_fn():
                return else_expression
        else:
            else_expression_fn = else_expression
        x = tf.cond(if_condition, 
                    then_expression_fn,
                    else_expression_fn)
        return x
        
    # RANDOM GENERATORS
    def rng_normal(self, shape, mean=0., std=1, dtype=None, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        dtype = _str_dtype(dtype)
        return tf.random_normal(shape, mean=mean, stddev=std,
                                dtype=dtype, seed=seed)
        
    def rng_truncated_normal(self, shape, mean=0., std=1., dtype=None, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        dtype = _str_dtype(dtype)
        return tf.truncated_normal(shape, mean, std, dtype=dtype, seed=seed)
        
    def rng_uniform(self, shape, low=0., high=1., dtype=None, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        dtype = _str_dtype(dtype)
        return tf.random_uniform(shape, minval=low, maxval=high,
                                 dtype=dtype, seed=seed)
        
    def rng_binomial(self, shape, p=0., dtype=None, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        dtype = _str_dtype(dtype)
        return tf.select(tf.random_uniform(shape=shape, dtype=dtype, 
                                    seed=seed) <= p,
                                    tf.ones(shape, dtype=dtype),
                                    tf.zeros(shape, dtype=dtype))
                                    
    #NeuralNet OPS
    def _pre_conv2d_input(self, x, dim_ordering):
        if(self._default_dtype == 'float64'):
            x = tf.cast(x, 'float32')
        if(dim_ordering == 'th'):
            #TF uses the last dimension as the channel dimension
            #instead of the second one
            #theano input shape (samples, input_depth, rows, columns)
            #TF input shape (samples, rows, columns, depth)
            x = tf.transpose(x, (0, 2, 3, 1))
        return x
        
    def _pre_conv2d_kernel(self, kernel, dim_ordering):
        if(self._default_dtype == 'float64'):
            kernel = tf.cast(kernel, 'float32')
        if(dim_ordering == 'th'):
            #Theano kernel ordering (depth, input_depth, rows, cols)
            #TF kernel ordering (rows, cols, input_depth, depth)
            kernel = tf.transpose(kernel, (2, 3, 1, 0))
        return kernel
        
    def _pre_bordermode(self, border_mode):
        if(border_mode == 'same'):
            padding = 'SAME'
        elif(border_mode == 'valid'):
            padding = 'VALID'
        else:
            raise Exception("Invalid border mode {}".format(border_mode))
        return padding
        
    def _post_conv2d_output(self, x, dim_ordering):
        if(dim_ordering == 'th'):
            x = tf.transpose(x, (0, 3, 1, 2))
        if(self._default_dtype == "float64"):
            x = tf.cast(x, 'float64')
        return x
        
    def temporal_padding(self, x, padding=(1, 1)):
        """Pads the middle dimension of a 3D tensor.
        # Arguments
            x: Tensor or variable.
            padding: Tuple of 2 integers, how many zeros to
                add at the start and end of dim 1.
        # Returns
            A padded 3D tensor.
        """
        assert len(padding) == 2
        pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
        return tf.pad(x, pattern)
        
    def conv1d(self, x, kernel, strides=1, padding='valid',
               dilation_rate=1):
        kernel_shape = kernel.get_shape().as_list()
        if(padding == 'causal'):
            left_pad = dilation_rate * (kernel_shape[0] - 1)
            x = self.temporal_padding(x, (left_pad, 0))
            padding = 'valid'
        
        if(padding == 'same'):
            padding = 'SAME'
        elif(padding == 'valid'):
            padding = 'VALID'
        else:
            raise ValueError('Invalid padding')
        
        x = tf.nn.convolution(input=x,
                              filter=kernel,
                              dilation_rate=(dilation_rate,),
                                strides=(strides,),
                              padding=padding,
                              data_format='NCW')
        return x

    def conv2d(self, x, kernel, strides=(1, 1), padding='valid',
               dilation_rate=(1, 1)):
        x = self._pre_conv2d_input(x, 'th')
        kernel = self._pre_conv2d_kernel(kernel, 'th')
        if(padding == 'same'):
            padding = 'SAME'
        elif(padding == 'valid'):
            padding = 'VALID'
        else:
            raise ValueError('Invalid padding')
        
        x = tf.nn.convolution(
                              input=x,
                              filter=kernel,
                              dilation_rate=dilation_rate,
                              strides=strides,
                              padding=padding,
                              data_format='NHWC')
        return self._post_conv2d_output(x, 'th')
        
    def pool2d(self, x, pool_size, strides=(1, 1), padding='valid',
               pool_mode='max'):
        if(padding == 'same'):
            padding = 'SAME'
        elif(padding == 'valid'):
            padding = 'VALID'
        else:
            raise ValueError('Invalid padding')
            
        strides = (1, ) + strides + (1, )
        pool_size = (1, ) + pool_size + (1, )
        x = self._pre_conv2d_input(x, 'th')
        if(pool_mode == 'max'):
            x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
        elif(pool_mode == 'avg'):
            x = tf.nn.avg_pool(x, pool_size, strides, padding=padding)
        else:
            raise ValueError("Unknown pool_mode {}".format(pool_mode))
        return self._post_conv2d_output(x, 'th')
        
    def reverse(self, x, axes):
        if(isinstance(axes, int)):
            axes = [axes]
        return tf.reverse(x, axes)
        
    def recurrence(self, step_function, inputs, initial_states,
            unroll=False,
            go_backwards=False, constants=None, input_length=None):
        ndim = len(inputs.get_shape())
        if(ndim < 3):
            raise ValueError("Input should be at least 3D")
        axes = [1, 0] + list(range(2, ndim))
        inputs = tf.transpose(inputs, (axes))
        
        if(constants is None):
            constants = []
            
        if(unroll):
            if(not inputs.get_shape()[0]):
                raise ValueError("Unrolling requires a fixed "
                                 "number of timesteps")
            states = initial_states
            successive_states = []
            successive_outputs = []
            
            input_list = tf.unstack(inputs)
            if(go_backwards):
                input_list.reverse()
            for inp in input_list:
                output, states = step_function(inp, states+constants)
                successive_outputs.append(output)
                successive_states.append(states)
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = tf.stack(successive_outputs)
        
        else:
            if(go_backwards):
                inputs = self.reverse(inputs, 0)
            
            states = tuple(initial_states)
            time_steps = tf.shape(inputs)[0]
            outputs, _ = step_function(inputs[0], initial_states+constants)
            output_ta = tensor_array_ops.TensorArray(
                                                     dtype=outputs.dtype,
                                                     size=time_steps,
                                                     tensor_array_name='output_ta')
            input_ta = tensor_array_ops.TensorArray(
                                                    dtype=inputs.dtype,
                                                    size=time_steps,
                                                    tensor_array_name='input_ta')
            input_ta = input_ta.unstack(inputs)
            time = tf.constant(0, dtype='int32', name='time')
            
            def _step(time, output_ta_t, *states):
                """
                RNN step function
                # Args
                @param time: Current timestep
                @param output_ta_t: TensorArray
                *states: List of states
                
                # Returns
                  tuple: (time+1, output_ta_t) + tuple(new_states)
                """
                current_input = input_ta.read(time)
                output, new_states = step_function(current_input,
                                                   tuple(states)+
                                                   tuple(constants))
                for state, new_state in zip(states, new_states):
                    new_state.set_shape(state.get_shape())
                output_ta_t = output_ta_t.write(time, output)
                return (time + 1, output_ta_t) + tuple(new_states)
                
            final_outputs = control_flow_ops.while_loop(
                    cond=lambda time, *_: time < time_steps,
                    body=_step,
                    loop_vars=(time, output_ta) + states,
                    parallel_iterations=32,
                    swap_memory=True)
            last_time = final_outputs[0]
            output_ta = final_outputs[1]
            new_states = final_outputs[2:]
            
            outputs = output_ta.stack()
            last_output = output_ta.read(last_time - 1)
        
        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        return last_output, outputs, new_states
            

    def l2_norm(self, x, axis):
        if(axis < 0):
            axis = axis % len(x.get_shape())
        return tf.nn.l2_normalize(x, dim=axis)
        
    def dropout(self, x, probability):
        retain = 1. - probability
        return tf.nn.dropout(x * 1., retain, seed=self._rng_seed)
        
    def tanh(self, x):
        return tf.nn.tanh(x)
        
    def hard_sigmoid(self, x):
        x = (0.2 * x) + 0.5
        return self.clip(x, 0., 1.)
        
    def sigmoid(self, x):
        return tf.nn.sigmoid(x)
        
    def binary_crossentropy(self, output, target, logit=False):
        if(not logit):
            output = self.clip(output, self._epsilon, 1 - self._epsilon)
            output = tf.log(output / (1 - output))
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=output, 
                                                       labels=target)
            
        
    def categorical_crossentropy(self, output, target, logit=False):
        if(not logit):
            output /= tf.reduce_sum(output, reduction_indices=len(output.get_shape())-1,
                                    keep_dims=True)
            output = self.clip(output, self._epsilon, 1. - self._epsilon)
            return -tf.reduce_sum(target * tf.log(output), 
                                  reduction_indices=len(output.get_shape()) - 1)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(logits=output, 
                                                           labels=target)
            
        
    def softmax(self, x):
        return tf.nn.softmax(x)
        
    def softplus(self, x):
        return tf.nn.softplus(x)
        
    def relu(self, x, alpha, max_value=None):
        if(alpha != 0.):
            neg = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        if(max_value is not None):
            x = self.clip(x, 0., max_value)
        if(alpha != 0.):
            alpha = self.to_tensor(alpha, x.dtype.base_dtype)
            x -= alpha * neg
        return x
        
        

if __name__ == "__main__":
    N = TensorflowBackend()
    arr = numpy.random.randn(500)
    v = N.rng_binomial(shape=arr.shape, p=0.8)
    f = N.function([], v)
    