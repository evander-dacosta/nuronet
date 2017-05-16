# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:01:22 2016

@author: evander
"""

import numpy
from nuronet2.backend import N, get_from_module
from scipy.stats import truncnorm


def get_weightfactory(name):
    return get_from_module(name, globals(), "weightfactory",
                           instantiate=True)
                           

def zeros():
    return Zeros()

def ones():
    return Ones()
    
def constant(value):
    return Constant(value=value)
    
def normal(mean=0., std=0.01):
    #return NormalWeights(std=std, mean=mean)
    return RandomNormal(std=std, mean=mean)
    
def truncated_normal(mean=0., std=0.01):
    return TruncatedNormal(mean=mean, std=std)
    
def uniform(mean=0., minval=-0.01, maxval=0.01):
    #return UniformWeights(mean=mean, range=range)
    return RandomUniform(minval=minval, maxval=maxval)
    
def orthogonal(gain=1.):
    return Orthogonal(gain=gain)
    
def lecun_uniform():
    """
    draws samples from uniform distribution within [-limit, limit]
    where limit = sqrt(3 / fan_in)
    """
    return VarianceScaling(scale=1., mode='fan_in', distribution='uniform')
    
def xavier_normal(is_convolution=False):
    """
    draws samples from a truncated normal distribution centered at 0.
    The std is sqrt(2 / (fan_in + fan_out))
    """
    return VarianceScaling(scale=1., mode='fan_avg', distribution='normal')
    
def xavier_uniform(is_convolution=False):
    """
    Draws samples from a uniform distribution within [-limit, limit]
    where limit = sqrt(6 / fan_in + fan_out)
    """
    return VarianceScaling(scale=1., mode='fan_avg', distribution='uniform')
    
def he_normal(is_convolution=False):
    """
    Draws samples from a truncated normal distribution centered on 0.
    std = sqrt(2 / fan_in)
    """
    return VarianceScaling(scale=2., mode='fan_in', distribution='normal')
    
def he_uniform(is_convolution=False):
    """
    Draws samples from a uniform distribution within [-limit, limit]
    limit is sqrt(6 / fan_in)
    """
    return VarianceScaling(scale=2., mode='fan_in', distribution='uniform')
    
    
    
def _compute_fans(shape):
    if(len(shape) == 2):
        fan_in = shape[0]
        fan_out = shape[1]
    elif(len(shape) in [3, 4, 5]):
        receptive_field_size = numpy.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in = numpy.sqrt(numpy.prod(shape))
        fan_out = numpy.sqrt(numpy.prod(shape))
    return fan_in, fan_out

def _truncated_normal_vars(mean, std, shape, seed=None):
    if(seed is None):
        seed = numpy.random.randint(1e7)
    numpy.random.seed(seed=seed)
    weights = truncnorm(a=mean-2*std, b=mean+2*std).rvs(shape)
    weights = weights.astype(N.floatx)
    return weights

class WeightFactory(object):

    """
    Base class for initialising tensor weights/biases
    """

    def __call__(self, shape, name=None):
        return self.make_weights(shape, name)

    def make_weights(self, shape, name):
        """
        Has to be reimplemented.

        Must return a shared variable of type theano.config.floatX
        """
        raise NotImplementedError()

    
class Zeros(WeightFactory):
    def make_weights(self, shape, name):
        return N.shared(numpy.zeros(shape=shape, dtype=N.floatx), name=name)

class Ones(WeightFactory):
    def make_weights(self, shape, name):
        return N.shared(numpy.ones(shape=shape, dtype=N.floatx), name=name)

    
class Constant(WeightFactory):
    def __init__(self, value=0.):
        self.value = value
        
    def make_weights(self, shape, name):
        constant = self.value * numpy.ones(shape=shape, dtype=N.floatx)
        return N.shared(constant, name=name)
        
        
        
class RandomNormal(WeightFactory):
    def __init__(self, mean=0., std=0.05, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        self.mean = mean
        self.std = std
        self.seed = seed
        
    def make_weights(self, shape, name):
        weights = numpy.random.RandomState(seed=self.seed).normal(self.mean,
                                                        self.std, size=shape)
        weights = weights.astype(N.floatx)
        return N.shared(weights, name=name)
        
        
class RandomUniform(WeightFactory):
    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        
    def make_weights(self, shape, name):
        weights = numpy.random.RandomState(seed=self.seed).uniform(low=self.minval,
                                                high=self.maxval, size=shape)
        weights = weights.astype(N.floatx)
        return N.shared(weights, name=name)
        


class TruncatedNormal(WeightFactory):
    """
    Just like normal weight factory, except values that are
    more than two stds from the mean are discarded and redrawn
    """
    def __init__(self, mean=0., std=0.05, seed=None):
        if(seed is None):
            seed = numpy.random.randint(1e7)
        self.mean = mean
        self.std = std
        self.seed = seed

    def make_weights(self, shape,name):
        weights = _truncated_normal_vars(mean=self.mean, std=self.std, 
                                         shape=shape, seed=self.seed)
        return N.shared(weights, name=name)
        
        
        
class VarianceScaling(WeightFactory):
    """
    A weight factory that scales itself to the shape of weights
    
    # Arguments
        scale: A positive scaling factor
        mode: One of 'fan_in', 'fan_out', and 'fan_avg'
        distribution: Random distribution. One of 'normal' or 'uniform'
        seed: RNG seed
    """
    def __init__(self, scale=1., mode='fan_avg', distribution='normal',
                 seed=None):
        if(scale < 0.):
            raise ValueError("scale needs to be a positive float. Given", scale)
        mode = mode.lower()
        if(mode not in ['fan_in', 'fan_out', 'fan_avg']):
            raise ValueError("Unexpected mode argument. Needs one of "
                             "['fan_in', 'fan_out', 'fan_avg']. Got", mode)
        distribution = distribution.lower()
        if(distribution not in ['normal', 'uniform']):
            raise ValueError("Unexpected distribution argument. Needs one of "
                             "['normal', 'uniform']. Got", mode)
        if(seed is None):
            seed = numpy.random.randint(1e7)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed
        
    def make_weights(self, shape, name):
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if(self.mode == 'fan_in'):
            scale /= max(1., fan_in)
        elif(self.mode == 'fan_out'):
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2.)
        
        if(self.distribution == 'normal'):
            std = numpy.sqrt(scale)
            weights = _truncated_normal_vars(mean=0., std=std, shape=shape,
                                             seed=self.seed)
            return N.shared(weights, name=name)
        else:
            limit = numpy.sqrt(3. * scale)
            weights = numpy.random.RandomState(seed=self.seed).uniform(-limit,
                                                        limit, size=shape)
            weights = weights.astype(N.floatx)
            return N.shared(weights, name=name)

class Orthogonal(WeightFactory):
    """
    Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    def __init__(self, gain=1.):
        self.gain = gain
        
    def make_weights(self, shape, name=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        seed = numpy.random.randint(1e7)
        numpy.random.seed(seed)
        a = numpy.random.normal(0., 1., flat_shape)
        u, _, v = numpy.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q.reshape(shape)
        q = q.astype(N.floatx)
        q = self.gain * q[:shape[0], :shape[1]]
        return N.shared(q, name=name)
