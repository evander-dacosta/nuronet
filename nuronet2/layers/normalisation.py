#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:19:27 2017

@author: evander
"""

import numpy
from nuronet2.backend import N
from nuronet2.base import get_weightfactory, get_regulariser, Layer

class BatchNormalisation(Layer):
    """
    Normalise the activations of the previous layer for each
    minibatch. This helps reduce internal covariate shift, increasing
    training speed.
    
    # Arguments
        axis: The axis that should be normalised (Typically a features axis)
              For example, for a Conv2d layer (batches, filters, height, width),
              set axis = 1
              
        momentum: Momentum for calculating moving averages
        
        epsilon: Small float to avoid divide-by-zero
        
        center: If True, add beta to the normalised tensor
        
        scale: If True, multiply by gamma. If the next layer if linear
                e.g. relu, then turn this off because the scaling 
                will be done in the next layer
                
        beta_factory: Weight factory for beta weight
        
        gamma_factory: Weight factory for gamma weight
        
        moving_mean_factory: Weight factory for moving mean
        
        moving_variance_factory: Weight factory for moving vaciranve
    """
    
    def __init__(self, axis=1, momentum=0.99, epsilon=1e-3,
                 center=True, scale=True, beta_factory='zeros',
                 gamma_factory='ones', moving_mean_factory='zeros',
                 moving_variance_factory='ones', **kwargs):
        super(BatchNormalisation, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_factory = get_weightfactory(beta_factory)
        self.gamma_factory = get_weightfactory(gamma_factory)
        self.moving_mean_factory = get_weightfactory(moving_mean_factory)
        self.moving_variance_factory = get_weightfactory(moving_variance_factory)
        
    def build(self, input_shape):
        dim = input_shape[self.axis]
        if(dim is None):
            raise ValueError('Axis ' + self.axis + ' of '
                            'input_tensor should have a definite dimension '
                            'instead got shape ' + str(input_shape))
        shape = (dim, )
        if(self.scale):
            self.gamma = self.gamma_factory(shape=shape, name='gamma')
        else:
            self.gamma = None
            
        if(self.center):
            self.beta = self.beta_factory(shape=shape, name='beta')
        else:
            self.beta = None
            
        self.moving_mean = self.moving_mean_factory(shape=shape,
                                                    name='moving_mean')
        self.moving_variance = self.moving_variance_factory(shape=shape,
                                                            name='moving_variance')
        self.trainable_weights = [self.gamma, self.beta]
        self.built = True
        
    def prop_up(self, x):
        input_shape = N.int_shape(x)
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])
        
        def normalise_inference():
            if(needs_broadcasting):
                broadcast_moving_mean = N.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = N.reshape(self.moving_variance,
                                                      broadcast_shape)
                if(self.center):
                    broadcast_beta = N.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if(self.scale):
                    broadcast_gamma = N.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return N.batch_normalization(
                        x,
                        broadcast_moving_mean,
                        broadcast_moving_variance,
                        broadcast_beta,
                        broadcast_gamma,
                        epsilon=self.epsilon)
            else:
                return N.batch_normalization(
                        x,
                        self.moving_mean,
                        self.moving_variance,
                        self.beta,
                        self.gamma,
                        epsilon=self.epsilon)
                        
        if(not self.is_training):
            return normalise_inference()
            
        normed_training, mean, variance = N.normalize_batch_in_training(
            x, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)
            
        def make_moving_average_update(variable, value, momentum):
            momentum = numpy.asarray([momentum], dtype=N.floatx)
            return (variable, variable * momentum[0] + value * (1. - momentum)[0])
        
        self.add_update(make_moving_average_update(self.moving_mean,
                                                   mean, self.momentum))
        self.add_update(make_moving_average_update(self.moving_variance, 
                                                   variance, self.momentum))
                
        if(self.is_training):
            return normed_training
        else:
            return normalise_inference()
    
    def get_cost(self):
        return 0.
            
    def get_output_shape(self, input_shape):
        return input_shape