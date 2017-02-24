# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:27:54 2017

@author: Evander
"""

import numpy
import time
import json
import warnings

from collections import deque, OrderedDict, Iterable
from nuronet2.backend import N, Progbar


class CallbackList(object):
    """
    Holds a list of callbacks
    """
    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        
    def append(self, callback):
        self.callbacks.append(callback)
        
    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)
            
    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)
            
    def epoch_start(self, epoch, logs=None):
        """
        Called right before an epoch starts
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.epoch_start(epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)
        
    def epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.epoch_end(epoch, logs)
            
    def batch_start(self, batch, logs=None):
        """
        Called right before processing a batch
        """
        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.batch_start(batch, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = numpy.median(self._delta_ts_batch_begin)
        if(self._delta_t_batch > 0. and 
          delta_t_median > 0.95 * self._delta_t_batch and
          delta_t_median > 0.1):
              warnings.warn("batch_start() is slow compared to the batch update "
                            ". Check your callbacks")
        self._t_enter_batch = time.time()
        
    def batch_end(self, batch, logs=None):
        """
        Called at the end of processing a batch
        """
        logs = logs or {}
        if(not hasattr(self, '_t_enter_batch')):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.batch_end(batch, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = numpy.median(self._delta_ts_batch_end)
        if(self._delta_t_batch > 0. and 
           (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)
                          
    def train_start(self, logs=None):
        """
        Called at the start of training
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.train_start(logs)
            
    def train_end(self, logs=None):
        """
        Called at the end of training
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.train_end(logs)
    
class Callback(object):
    """
    Base class for defining callbacks
    """
    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def epoch_start(self, epoch, logs=None):
        pass

    def epoch_end(self, epoch, logs=None):
        pass

    def batch_start(self, batch, logs=None):
        pass

    def batch_end(self, batch, logs=None):
        pass

    def train_start(self, logs=None):
        pass

    def train_end(self, logs=None):
        pass
    
    
class BaseLogger(Callback):
    """
    Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Nuronet model.
    """

    def epoch_start(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def batch_start(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def epoch_end(self, epoch, logs=None):
        print "epoch {}, loss {}, valid:{}".format(epoch, logs['loss'],
                                                logs['valid_loss'])
                    

            
class History(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def train_start(self, logs=None):
        self.epoch = []
        self.history = {}

    def epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
