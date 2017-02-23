# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:58:18 2016

@author: evander
"""

import numpy
from collections import OrderedDict

from nuronet2.base import Optimiser
from callbacks import AnnealedLearningRate, MomentumAdjust
from nuronet2.backend import N


def clip_norm(g, clip, n):
    return N.switch(N.ge(n, clip), g*clip/n, g)

def clip_grad(grad, clip = 0):
    if(clip > 0):
        norm = N.sqrt(sum([N.sum(g**2) for g in grad]))
        return [clip_norm(g, clip, norm) for g in grad]
    return grad


def compute_grads(lossOrGrad, params):
    if(isinstance(lossOrGrad, list)):
        if(not len(lossOrGrad) == len(params)):
            raise ValueError(
                "Got {} gradients for {} parameters.".format(
                    len(lossOrGrad),
                    len(params)) +
                " The two numbers should be equal")
        return lossOrGrad
    else:
        return N.gradients(lossOrGrad, params)


def sgd(loss, params, learningRate, weightDecay, clip):
    grads = compute_grads(loss, params)
    grads = clip_grad(grads, clip)
    updates = OrderedDict()

    for param, gParam in zip(params, grads):
        updates[param] = param - (learningRate * gParam) - \
                                        (learningRate * weightDecay * param)
    return updates
    
    
class SGD(Optimiser):

    """
        Performs Stochastic Gradient descent with a learning rate
        multiplier on the gradient at each iteration.
    """

    def __init__(self, model, objectives, lr_start, lr_stop=None, 
                 weight_decay = 0., clip = 0):
        """
        TODO:DOC
        """
        Optimiser.__init__(self, model=model, objectives=objectives)
        if(lr_stop is not None):
            lrAdjust = AnnealedLearningRate(start=lr_start, stop=lr_stop)
            self.add_callback(lrAdjust)
        self.learning_rate = N.cast(lr_start)
        self.weight_decay = N.cast(weight_decay)
        self.clip = clip

    def get_updates(self, cost, params):
        return sgd(cost, params, self.learning_rate, self.weight_decay, self.clip)
        
        

class Momentum(Optimiser):

    """
        TODO:DOC
    """

    def __init__(self, model, objectives, mom_start, lr_start,
                 mom_stop=None, lr_stop=None,
                 weight_decay = 0., clip = 0):
        Optimiser.__init__(self, model=model, objectives=objectives)
        assert mom_start >= 0.
        assert mom_start < 1.
        if(mom_stop is not None):
            assert mom_stop >= 0.
            assert mom_stop < 1.
            momCallback = MomentumAdjust(start=mom_start, stop=mom_stop)
            momCallback.setOptimiser(self)
            self.add_callback(momCallback)
        if(lr_stop is not None):
            lrCallback = AnnealedLearningRate(start=lr_start, stop=lr_stop)
            lrCallback.setOptimiser(self)
            self.add_callback(lrCallback)
        
        self.weightDecay = weight_decay
        self.clip = clip
        self.learning_rate = lr_start
        self.set_momentum(mom_start)

    def set_momentum(self, momentum):
        self.momentum = N.shared(momentum)

    def get_updates(self, cost, params):
        updates = sgd(cost, params, self.learning_rate, self.weightDecay, self.clip)
        for param in params:
            value = N.get_value(param)
            velocity = N.shared(
                numpy.zeros(
                    value.shape,
                    dtype=value.dtype))
            x = (self.momentum * velocity) + updates[param]
            updates[velocity] = x - param
            updates[param] = x
        return updates
        
        
class Nesterov(Momentum):

    def __init__(self, model, objectives, mom_start, lr_start, mom_stop=None, lr_stop=None,
                 weight_decay = 0., clip = 0):
        """
            TODO:DOC
        """
        Momentum.__init__(self, model, objectives, mom_start=mom_start, mom_stop=mom_stop,
                          lr_start=lr_start, lr_stop=lr_stop,
                          weight_decay = weight_decay, clip = clip)

    def get_updates(self, cost, params):
        updates = sgd(cost, params, self.learning_rate, self.weightDecay, self.clip)
        for param in params:
            value = N.get_value(param)
            velocity = N.shared(
                numpy.zeros(
                    value.shape,
                    dtype=value.dtype))
            x = (self.momentum * velocity) + updates[param] - param
            updates[velocity] = x
            updates[param] = (self.momentum * x) + updates[param]
        return updates
        
        
class Adagrad(Optimiser):
    def __init__(self, model, objectives, lr_start, lr_stop=None, epsilon=N._epsilon):
        Optimiser.__init__(self, model=model, objectives=objectives)
        if(lr_stop is not None):
            lrAdjust = AnnealedLearningRate(start=lr_start, stop=lr_stop)
            lrAdjust.setOptimiser(self)
            self.add_callback(lrAdjust)
        self.learning_rate = lr_start
        self.epsilon = N.cast(epsilon)
        
    def get_updates(self, cost, params):
        grads = compute_grads(cost, params)
        updates = OrderedDict()
        
        for param, grad in zip(params, grads):
            value = N.get_value(param)
            accumulate = N.shared(numpy.zeros(value.shape, dtype=value.dtype))
            newAccumulate = accumulate + grad ** 2
            updates[accumulate] = newAccumulate
            updates[param] = param - (self.learning_rate * grad /
                                      N.sqrt(newAccumulate + self.epsilon))
        return updates
        
        
class Adadelta(Optimiser):
    def __init__(self, model, objectives, lr_start=1.0, rho=0.95, epsilon=N._epsilon,
                 lr_stop=None):
        Optimiser.__init__(self, model=model, objectives=objectives)
        if(lr_stop is not None):
            lrAdjust = AnnealedLearningRate(start=lr_start, stop=lr_stop)
            lrAdjust.setOptimiser(self)
            self.add_callback(lrAdjust)
        self.learning_rate = N.cast(lr_start)
        self.epsilon = N.cast(epsilon)
        self.rho = N.cast(rho)
        
    def get_updates(self, cost, params):
        grads = compute_grads(cost, params)
        updates = OrderedDict()
        
        for param, grad in zip(params, grads):
            value = N.get_value(param)
            accum = N.shared(numpy.zeros(value.shape, dtype=value.dtype))
            deltaAccum = N.shared(numpy.zeros(value.shape, dtype=value.dtype))
            newAccum = (self.rho * accum) + ((1 - self.rho) * N.square(grad))
            updates[accum] = newAccum
            
            pUpdate = (grad * N.sqrt(deltaAccum + self.epsilon) /
                       N.sqrt(newAccum + self.epsilon))
            updates[param] = param - (self.learning_rate * pUpdate)
            
            newDeltaAccum = (self.rho * deltaAccum) + ((1 - self.rho) * N.square(pUpdate))
            updates[deltaAccum] = newDeltaAccum
        return updates
        
        
class Adam(Optimiser):
    def __init__(self, model, objectives, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=N._epsilon,
                 **kwargs):
        Optimiser.__init__(self, model=model, objectives=objectives)
        self.iterations = N.shared(0.)
        self.lr = N.shared(lr)
        self.beta_1 = N.shared(beta_1)
        self.beta_2 = N.shared(beta_2)
        self.epsilon = N.cast(epsilon)
        
    def get_updates(self, cost, params):
        grads = compute_grads(cost, params)
        updates = OrderedDict()
        updates[self.iterations] = self.iterations + 1
        
        t = self.iterations + 1
        lr_t = self.lr * N.sqrt(1. - N.pow(self.beta_2, t)) / (1. - N.pow(self.beta_1, t))
        
        batch_params = [N.get_value(x) for x in params]
        shapes = [x.shape for x in batch_params]
        ms = [N.zeros(shape) for shape in shapes]
        vs = [N.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs
        
        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * N.square(g)
            p_t = p - lr_t * m_t / (N.sqrt(v_t) + self.epsilon)
            updates[m] = m_t
            updates[v] = v_t
            updates[p] = p_t
        return updates