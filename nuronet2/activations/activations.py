# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:04:26 2016

@author: Evander
"""

from nuronet2.backend import N, get_from_module
import six
import types
import inspect

"""
TODO: FIND A WAY TO MAKE A LIST OF AVAILABLE ACTIVATIONS
"""


def softmax(x):
    ndim = N.ndim(x)
    if(ndim == 2):
        return N.softmax(x)
    elif(ndim == 3):
        ex = N.exp(x - N.max(x, axis=-1, keepdims=True))
        su = N.sum(ex, axis=-1, keepdims=True)
        return ex/su
    else:
        raise Exception("Cannot apply a softmax transformation to an {}-dimensional tensor".format(ndim))
        
def softplus(x):
    return N.softplus(x)
    
def relu(x, alpha=0., max_value=None):
    return N.relu(x, alpha=alpha, max_value=max_value)
    
def tanh(x):
    return N.tanh(x)
    
def tanh2(x):
    activation = 1.1759 * N.tanh(2*x / 3.)
    return N.cast(activation, N.default_dtype)
    
def sigmoid(x):
    return N.sigmoid(x)
    
def hard_sigmoid(x):
    return N.hard_sigmoid(x)
    
def linear(x):
    return x
    

def get_activation(name):
    if(isinstance(name, types.FunctionType)):
        if(len(inspect.getargspec(name).args) is not 1):
            raise Exception("A function was passed to get_activation, but " + \
                            "wasn't the right format to be an objective." + \
                            "Define objective functions as func(target, output)")
        return name
    return get_from_module(name, globals(), "activation",
                           instantiate=False)
