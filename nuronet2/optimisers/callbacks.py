# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:56:42 2016

@author: evander
"""

import numpy
from nuronet2.base import Optimiser


class Callback(object):

    def setOptimiser(self, optimiser):
        """
        Sets the optimiser which will be calling this CallBack
        at the end of each EPOCH
        """
        if(not isinstance(optimiser, Optimiser)):
            raise Exception("Callback requires an Optimiser type")
        self.optimiser = optimiser

    def __call__(self):
        assert(hasattr(self, 'optimiser'))
        self.call_function()


class AnnealedLearningRate(Callback):

    def __init__(self, start=0.03, stop=0.01):
        self.start = start
        self.stop = stop

    def call_function(self):
        if(not hasattr(self, 'ls')):
            self.ls = numpy.linspace(
                self.start,
                self.stop,
                self.optimiser.nEpochs)
        epoch = self.optimiser._current_epoch
        newValue = self.ls[epoch - 1]
        self.optimiser.learning_rate = newValue


class MomentumAdjust(Callback):

    def __init__(self, start=0.8, stop=0.999):
        self.start = start
        self.stop = stop

    def call_function(self):
        if(not hasattr(self, 'ls')):
            self.ls = numpy.linspace(
                self.start,
                self.stop,
                self.optimiser.nEpochs)
        epoch = self.optimiser._current_epoch
        newValue = self.ls[epoch - 1]
        self.optimiser.set_momentum(newValue)