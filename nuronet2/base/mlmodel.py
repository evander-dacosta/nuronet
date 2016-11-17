# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:54:21 2016

@author: evander
"""

from backend import N
from collections import OrderedDict

class MLModel(object):
    def __init__(self, input_shape, input=None, output=None,
                 name=None):
        self.input = input if input is not None else N.variable(ndim=len(input_shape))
        self.output = output
        self.input_shape = input_shape
        self.name = name
        self.is_built = False
        self._predictor = None
        self.is_train = False
        
    def get_predictor(self):
        return N.function([self.input], self())
        
    def set_train_mode(self, boolean):
        if(not isinstance(boolean, bool)):
            raise Exception("set_train_mode only takes boolean values." + \
                            " Given {}".format(type(boolean)))
        self.is_train = boolean
        
    @property
    def predictor(self):
        if(self._predictor is None):
            self._predictor = self.get_predictor()
        return self.predictor
        
    def __call__(self, x):
        if(isinstance(x, MLModel)):
            x = x.prop_up(x.input)
        return self.prop_up(x)
        
    #TO BE IMPLEMENTED
        
    def prop_up(self, *args, **kwargs):
        raise NotImplementedError()
        
    def build(self, *args, **kwargs):
        raise NotImplementedError()
        
    def get_cost(self):
        raise NotImplementedError()
        
    def get_params(self):
        raise NotImplementedError()
        
    def set_params(self, params):
        raise NotImplementedError()
        
    def get_output_shape(self):
        raise NotImplementedError()
        
    def get_updates(self):
        """
        Return custom updates
        """
        return OrderedDict()
        
        
if __name__ == "__main__":
    model = MLModel()
    model.set_train_mode(False)
    print model.is_train
    