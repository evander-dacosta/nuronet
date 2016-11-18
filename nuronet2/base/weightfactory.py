# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:01:22 2016

@author: evander
"""

import numpy
from nuronet2.backend import N, get_from_module


def get_weightfactory(name):
    return get_from_module(name, globals(), "weightfactory",
                           instantiate=True)
                           

    
    
def normal(mean=0., std=0.01):
    return NormalWeights(std=std, mean=mean)
    
def uniform(mean=0., range=0.01):
    return UniformWeights(mean=mean, range=range)
    
def xavier_normal(is_convolution=False):
    return XavierNormal(isConv=is_convolution)
    
def xavier_uniform(is_convolution=False):
    return XavierUniform(isConv=is_convolution)
    
def he_normal(is_convolution=False):
    return HeNormal(isConv=is_convolution)
    
def he_uniform(is_convolution=False):
    return HeUniform(isConv=is_convolution)
    
def orthogonal(scale=1.1):
    return Orthogonal(scale=scale)
    
def constant(const=1.):
    return ConstantWeights(const=const)
    


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
        
        
class NormalWeights(WeightFactory):

    """
    Returns weights that are normally distributed
    
    Parameters
    ----------
    std: standard deviation of the gaussian to sample weights from
    mean: mean of the gaussian to sample weights from
    """

    def __init__(self, std=0.01, mean=0.):
        self.std = std
        self.mean = mean

    def make_weights(self, shape, name):
        return N.shared(numpy.random.normal(self.mean, self.std, size=shape),
                        name=name)
        
class UniformWeights(WeightFactory):

    """
    Returns uniformly sampled weights from U(-range, range)

    Parameters
    ----------
    std: When std is given, weights are sampled from
        U(mean - sqrt(3.) * std, mean + sqrt(3.) * std)
    """

    def __init__(self, std=None, mean=0., range=0.01):
        if(std is not None):
            self.range = [
                mean - (numpy.sqrt(3.) * std), mean + (numpy.sqrt(3.) * std)]
        else:
            if(isinstance(range, (list, tuple))):
                self.range = range
            else:
                self.range = [-range, range]

    def make_weights(self, shape, name):
        return N.shared(numpy.random.uniform(low=self.range[0],
                                            high=self.range[1],
                                            size=shape),
                                            name=name)


class Xavier(WeightFactory):

    """

    Parameters
    ----------
    weightFactory: neuralnetwork.weightFactory.WeightFactory
                    used to sample the weights. Must accept std and mean
                    as init parameters

    gain : 'SIGMOID' or 'RELU'
            Scaling factor for weights. Set to sqrt(2) for rectified
            linear units. Else set to 1.0 for sigmoid units. Other
            transfer functions may need different gains

    isConv : bool
            Set to true if the calling layer is a convolutional net.
            This flag is required to set proper fan-in - fan-out


    References
    ----------
    Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
    """

    def __init__(self, weightFactory, gain=1.0, isConv=False):
        if(gain == 'RELU'):
            gain = numpy.sqrt(2)

        self.weightFactory = weightFactory
        self.gain = gain
        self.isConv = isConv

    def make_weights(self, shape, name):
        if(self.isConv):
            if(len(shape) != 4):
                raise RuntimeError(
                    "If 'isConv' is set to True for WeightFactory, " +
                    "only shapes of length 4 are accepted. Not length {}".format(
                        len(shape)))
        #shape is (features_current, features_below, filter_height, filter_width)
        nIn, nOut = shape[:2]
        receptiveFieldSize = numpy.prod(shape[2:])

        std = self.gain * numpy.sqrt(2. / ((nIn + nOut) * receptiveFieldSize))
        return self.weightFactory(std=std).make_weights(shape, name)


class XavierNormal(Xavier):

    def __init__(self, gain=1., isConv=False):
        #gain=numpy.sqrt(2.)
        Xavier.__init__(self, NormalWeights, gain, isConv)


class XavierUniform(Xavier):

    def __init__(self, gain=1., isConv=False):
        #gain=numpy.sqrt(6.)
        Xavier.__init__(self, UniformWeights, gain, isConv)


class He(WeightFactory):

    """
    Parameters
    ----------
    weightFactory: neuralnetwork.weightFactory.WeightFactory
                    used to sample the weights. Must accept std and mean
                    as init parameters

    gain : float or 'RELU'
            Scaling factor for weights. Set to sqrt(2) for rectified
            linear units. Else set to 1.0 for sigmoid units. Other
            transfer functions may need different gains

    isConv : bool
            Set to true if the calling layer is a convolutional net.
            This flag is required to set proper fan-in - fan-out

    References
    ----------
     [1] Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852.
    """

    def __init__(self, weightFactory, gain=1.0, isConv=False):
        if(gain == "RELU"):
            gain = numpy.sqrt(2)
        self.weightFactory = weightFactory
        self.gain = gain
        self.isConv = isConv

    def make_weights(self, shape, name):
        if(self.isConv):
            if(len(shape) != 4):
                raise RuntimeError(
                    "If 'isConv' is set to True for WeightFactory, " +
                    "only shapes of length 4 are accepted. Not length {}".format(
                        len(shape)))
            fanIn = numpy.prod(shape[1:])
        elif(len(shape) == 2):
            fanIn = shape[0]
        else:
            raise RuntimeError(
                "Weightfactory only works with shapes of length >=2")
        std = self.gain * numpy.sqrt(1. / fanIn)
        return self.weightFactory(std=std).make_weights(shape, name)


class HeNormal(He):

    def __init__(self, gain=1., isConv=False):
        He.__init__(self, NormalWeights, gain, isConv)


class HeUniform(He):

    def __init__(self, gain=1., isConv=False):
        He.__init__(self, UniformWeights, gain, isConv)
        
        
class Orthogonal(WeightFactory):
    """
    Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """
    def __init__(self, scale=1.1):
        self.scale = scale
        
    def make_weights(self, shape, name=None):
        flatShape = (shape[0], numpy.prod(shape[1:]))
        a = numpy.random.normal(0.0, 1.0, flatShape)
        u, _, v = numpy.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flatShape else v
        q = q.reshape(shape)
        return N.shared(self.scale * q[:shape[0], :shape[1]], name=name)


class ConstantWeights(WeightFactory):

    """
    Initialise constant weights
    """

    def __init__(self, const=1.0):
        self.const = const

    def make_weights(self, shape, name):
        return N.shared(numpy.ones(shape) * self.const, name=name)