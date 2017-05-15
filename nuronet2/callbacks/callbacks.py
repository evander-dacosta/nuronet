# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:27:54 2017

@author: Evander
"""

import numpy
import time
import json
import warnings
import sys
from tabulate import tabulate
from collections import OrderedDict

from collections import deque, OrderedDict, Iterable
from nuronet2.backend import N


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

    
class TrainLogger(Callback):
    """
    Prints metrics at the end of each epoch
    """
    def __init__(self, f=sys.stdout):
        self.print_headers = True
        self.f = f

    def epoch_end(self, epoch, logs):
        info_tabulate = OrderedDict([
            ('epoch', logs['epoch']),
            ('train_loss', '{:.5f}'.format(logs['train_loss']))
        ])
        if('valid_loss' in logs.keys()):
            info_tabulate['valid_loss'] = "{:.5f}".format(float(logs['valid_loss']))
        info_tabulate['epoch_time'] = "{:.2f} s".format(logs['epoch_time'])         
        tab = tabulate([info_tabulate], headers="keys", floatfmt='.5f')
        out = ""
        if(self.print_headers):
            out = "\n".join(tab.split('\n', 2)[:2])
            out += "\n"
            self.print_headers = False

        out += tab.rsplit("\n", 1)[-1]
        out += "\n"
        self.f.write(out)
                    

            
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

    
    def plot(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure()
        plt.plot(self.epoch, self.history['train_loss'], label='train_loss')
        plt.plot(self.epoch, self.history['valid_loss'], label='valid_loss')
        plt.legend()
        plt.show()
        
    def get_validation_scores(self):
        """
        Return a list of the loss scores on the validation set
        so we can evaluate bias and variance
        """
        return numpy.array(self.history['valid_loss'])
