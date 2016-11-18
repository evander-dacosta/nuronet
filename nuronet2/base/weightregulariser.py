# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:02:39 2016

@author: evander
"""

import six
from nuronet2.backend import N, get_from_module


def get_regulariser(name):
    return get_from_module(name, globals(), "regulariser",
                           instantiate=True)
    
        
def l1(l1=0.01):
    return WeightRegulariser(l1=l1)
    
def l2(l2=0.01):
    return WeightRegulariser(l2=l2)
    
def l1l2(l1=0.01, l2=0.01):
    return WeightRegulariser(l1=l1, l2=l2)
    


class Regulariser(object):
    def regularise(self, param):
        raise NotImplementedError()
        
        
    def __call__(self, param):
        return self.regularise(param)
        
        
class WeightRegulariser(Regulariser):
    def __init__(self, l1=0., l2=0.):
        self.l1 = N.cast(l1)
        self.l2 = N.cast(l2)
        
    def regularise(self, param):
        lOne = N.sum(N.abs(param)) * self.l1
        lTwo = N.sum(N.square(param)) * self.l2
        return lOne + lTwo