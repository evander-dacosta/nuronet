# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:30:47 2017

@author: Evander
"""
import numpy
from nuronet2.backend import N, get_from_module
from six.moves import zip


#TODO:
# 1) Test all of these
# 2) Create wrapper for tf's builtin optimisers
# update get

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

    

class Optimiser(object):
    """
    Base class for all optimisers
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if(k not in allowed_kwargs):
                raise ValueError("Unexpected keyword argument "
                                "passed to optimiser: {}".format(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []
        
    def get_updates(self, params, constraints, loss):
        raise NotImplementedError()
        
    def get_gradients(self, loss, params):
        grads = compute_grads(loss, params)
        if(hasattr(self, 'clipnorm') and self.clipnorm > 0):
            norm = N.sqrt(sum([N.sum(N.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if(hasattr(self, 'clipvalue') and self.clipvalue > 0):
            grads = [N.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads
        
    def set_weights(self, weights):
        """
        Sets the weights of the optimiser, from numpy arrays
        
        Inputs
        ------
            @param weights: a list of numpy arrays.
        """
        params = self.weights
        weight_value_tuples = []
        param_values = [N.get_value(param) for param in params]
        for pv, p, w in zip(param_values, params, weights):
            if(pv.shape != w.shape):
                raise ValueError("Optimiser weight shape " + \
                                str(pv.shape) + 
                                " not compatible with provided weight shape " +
                                str(w.shape))
            weight_value_tuples.append((p, w))
        for x, value in weight_value_tuples:
            N.set_value(x, numpy.asarray(value, dtype=N.dtype(x)))
            
    def get_weights(self):
        return [N.get_value(w) for w in self.weights]
        
    
class SGD(Optimiser):
    """
    Stochastic Gradient Descent Optimiser.
    
    Includes support for momentum, learning rate decay and Nesterov
    
    Inputs
    ------
        @param lr: A learning rate parameter
        @param momentum: momentum for parameter updates
        @decay: Learning rate decay over each update
        @nesterov: True/False whether to apply nesterov or not
    """
    
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        Optimiser.__init__(self, **kwargs)
        self.iterations = N.shared(0.)
        self.lr = N.shared(lr)
        self.momentum = N.shared(momentum)
        self.decay = N.shared(decay)
        self.initial_decay = decay
        self.nesterov = nesterov
        
    def get_updates(self, params, loss):
        grads = compute_grads(loss, params)
        self.updates = []
        
        lr = self.lr
        if(self.initial_decay > 0):
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append((self.iterations, self.iterations + 1))
            
        #momentum
        shapes = [N.shared_shape(p) for p in params]
        moments = [N.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = (self.momentum * m) - (lr * g) #velocity
            self.updates.append((m, v))
            
            if(self.nesterov):
                new_p = p + (self.momentum * v) - (lr * g)
            else:
                new_p = p + v
            
            self.updates.append((p, new_p))
        return self.updates
        
    
class RMSprop(Optimiser):
    """
    Implements RMSprop
    
    Inputs
    ------
        @param lr: float >= 0. Learning rate.
        @param rho: float >= 0.
        @param epsilon: float >= 0. Fuzz factor.
        @param decay: float >= 0. Learning rate decay over each update.
            
    """
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8, decay=0.,
                 **kwargs):
        Optimiser.__init__(self, **kwargs)
        self.lr = N.shared(lr)
        self.rho = N.shared(rho)
        self.epsilon = N.cast(epsilon)
        self.decay = N.shared(decay)
        self.initial_decay = decay
        self.iterations = N.shared(0.)
        
    def get_updates(self, params, loss):
        grads = compute_grads(loss, params)
        shapes = [N.shared_shape(p) for p in params]
        accumulators = [N.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = []
        
        lr = self.lr
        if(self.initial_decay > 0):
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append((self.iterations, self.iterations + 1))
        
        for p, g, a in zip(params, grads, accumulators):
            new_a = self.rho * a + (1. - self.rho) * N.square(g)
            self.updates.append((a, new_a))
            new_p = p - lr * g / (N.sqrt(new_a) + self.epsilon)
            
            self.updates.append((p, new_p))
        return self.updates
            
class Adagrad(Optimiser):
    """
    Implements Adagrad
    """
    def __init__(self, lr=0.01, epsilon=1e-8, decay=0., **kwargs):
        Optimiser.__init__(self, **kwargs)
        self.lr = N.shared(lr)
        self.epsilon = N.cast(epsilon)
        self.decay = N.shared(decay)
        self.initial_decay = decay
        self.iterations = N.shared(0.)
    
    def get_updates(self, params, loss):
        grads = compute_grads(loss, params)
        shapes = [N.shared_shape(p) for p in params]
        accumulators = [N.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = []
        
        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append((self.iterations, self.iterations + 1))
        
        for p, g, a in zip(params, grads, accumulators):
            new_a = a + N.square(g)
            self.updates.append((a, new_a))
            new_p = p - lr*g / (N.sqrt(new_a) + self.epsilon)
            self.updates.append((p, new_p))
        return self.updates
        
class Adadelta(Optimiser):
    """
    Implements Adadelta
    """
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-8, decay=0.,
                 **kwargs):
        Optimiser.__init__(self, **kwargs)
        self.lr = N.shared(lr)
        self.rho = N.cast(rho)
        self.epsilon = epsilon
        self.decay = N.shared(decay)
        self.initial_decay = decay
        self.iterations = N.shared(0.)
        
    def get_updates(self, params, loss):
        grads = compute_grads(loss, params)
        shapes = [N.shared_shape(p) for p in params]
        accumulators = [N.zeros(shape) for shape in shapes]
        delta_accum = [N.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accum
        self.updates = []
        
        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append((self.iterations, self.iterations + 1))
            
            
        for p, g, a, d_a in zip(params, grads, accumulators, delta_accum):
            new_a = self.rho * a + (1. - self.rho) * N.square(g)
            self.updates.append((a, new_a))
            
            update = g * N.sqrt(d_a + self.epsilon) / N.sqrt(new_a + self.epsilon)
            
            new_p = p - lr*update
            self.updates.append((p, new_p))
        return self.updates
        
class Adam(Optimiser):
    """
    Implements Adam optimiser
    """
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        Optimiser.__init__(self, **kwargs)
        self.iterations = N.shared(0)
        self.lr = N.shared(lr)
        self.beta_1 = N.shared(beta_1)
        self.beta_2 = N.shared(beta_2)
        self.epsilon = N.cast(epsilon)
        self.decay = N.shared(decay)
        self.initial_decay = decay
        
    def get_updates(self, params, loss):
        grads = compute_grads(loss, params)
        self.updates.append((self.iterations, self.iterations+1))
       
        lr = self.lr
        if(self.initial_decay > 0):
            lr *= (1. / (1. + self.decay * self.iterations))
        t = self.iterations + 1
        lr_t = lr * (N.sqrt(1. - N.pow(self.beta_2, t))) / (1. - N.pow(self.beta_1, t))
        
        shapes = [N.shared_shape(p) for p in params]
        ms = [N.zeros(shape) for shape in shapes]
        vs = [N.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs
        
        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * N.square(g)
            p_t = p - lr_t*m_t / (N.sqrt(v_t) + self.epsilon)
            
            self.updates.append((m, m_t))
            self.updates.append((v, v_t))
            
            new_p = p_t
            self.updates.append((p, new_p))
        return self.updates
       
    
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
            
def get_optimiser(identifier, **kwargs):
    if(isinstance(identifier, Optimiser)):
        return identifier
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, **kwargs)