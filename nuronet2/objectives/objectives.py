# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:06:38 2016

@author: Evander
"""

import numpy
import types
import inspect
from backend import N, get_from_module


def get_objective(name):
    if(isinstance(name, Objective)):
        return name
    if(isinstance(name, types.FunctionType)):
        if(len(inspect.getargspec(name).args) is not 2):
            raise Exception("A function was passed to get_objective, but " + \
                            "wasn't the right format to be an objective." + \
                            "Define objective functions as func(target, output)")
        objective = Objective()
        objective.get_error = name
        return objective
        
    #elif(isinstance(name, type) and isinstance(name(), Objective)):
    #    return name()
    return get_from_module(name, globals(), 'objective', instantiate=True)

def mean_squared_error():
    return MSE()
    
def mse2():
    return MSE2()
    
def mean_absolute_error():
    return MeanAbsoluteError()
    
def mean_percent_error():
    return MeanPercentError()
    
def square_hinge():
    return SquareHinge()
    
def hinge():
    return Hinge()
    
def categorical_crossentropy():
    return CategoricalXEntropy()
    
def binary_crossentropy():
    return BinaryXEntropy()
    

class Objective(object):
    def get_error(self, target, output):
        raise NotImplementedError()
        
    def __call__(self, target, output):
        return self.get_error(target, output)
        
        
class MSE(Objective):
    def get_error(self, target, output):
        return N.mean(N.square(target - output))
        
class MSE2(Objective):
    def get_error(self, target, output):
        num = N.sum(N.square(target - output))
        denom = N.sum(N.square(target))
        return num / denom
        
    
class MeanAbsoluteError(Objective):
    def get_error(self, target, output):
        return N.mean(N.abs(target - output))
        
class MeanPercentError(Objective):
    def get_error(self, target, output):
        diff = N.abs((target - output)/N.clip(N.abs(target), N._epsilon, numpy.inf))
        return 100. * N.mean(diff)
        
        
class SquareHinge(Objective):
    def get_error(self, target, output):
        return N.mean(N.square(N.maximum(1. - target * output, 0.)))
        
class Hinge(Objective):
    def get_error(self, target, output):
        return N.mean(N.maximum(1. - target * output, 0.))
        
        
class CategoricalXEntropy(Objective):
    def get_error(self, target, output):
        return N.mean(N.categorical_crossentropy(output, target))
        

class BinaryXEntropy(Objective):
    def get_error(self, target, output):
        return N.mean(N.binary_crossentropy(output, target))
        
        
if __name__ == "__main__":
    a = get_objective('mean_squared_error')