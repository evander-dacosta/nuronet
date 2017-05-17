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
        if('train_acc' in logs.keys()):
            info_tabulate['train_acc']="{:.5f}".format(float(logs['train_acc']))
        if('valid_acc' in logs.keys()):
            info_tabulate['valid_acc']="{:.5f}".format(float(logs['valid_acc']))
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

    
    def plot(self, metric='loss'):
        """
        Can plot loss / acc
        loss = loss
        acc = accuracy
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        if(metric.startswith('acc') or metric.startswith('err')):
            metric = metric[:3]
        if(metric not in ['acc', 'loss', 'err']):
            raise ValueError("Argument to History.plot() must be one of "
                             "'loss',  'acc' or 'err'. Given", metric)
        if(metric in ['acc', 'loss']):
            train_data = self.history['train_'+metric]
            valid_data = self.history['valid_'+metric]
        else:
            train_data = 1. - numpy.array(self.history['train_acc'])
            valid_data = 1. - numpy.array(self.history['valid_acc'])
        
        plt.figure()
        plt.plot(self.epoch, train_data, label='train_'+metric)
        plt.plot(self.epoch, valid_data, label='valid_'+metric)
        plt.legend()
        plt.show()
        
    def get_validation_scores(self):
        """
        Return a list of the loss scores on the validation set
        so we can evaluate bias and variance
        """
        return numpy.array(self.history['valid_loss'])
        
class Progbar(object):
    def __init__(self, target, width=30, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        
    def update(self, current, values=None, force=False):
        """
        current: Index of current step
        values: list of tuples (name, value_for_last_step). Progressbar
                will display averages for these values
        force: Whether or not to force visual progress updates
        """
        values = values or []
        for k, v in values:
            if(k not in self.sum_values):
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                     current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += (v*current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current
        
        now = time.time()
        if not force and (now - self.last_update) < self.interval:
            return

        prev_total_width = self.total_width
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')

        numdigits = int(numpy.floor(numpy.log10(self.target))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width - 1))
            if current < self.target:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
        sys.stdout.write(bar)
        self.total_width = len(bar)

        if current:
            time_per_unit = (now - self.start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.target - current)
        info = ''
        if current < self.target:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - self.start)
        for k in self.unique_values:
            info += ' - %s:' % k
            if isinstance(self.sum_values[k], list):
                avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                if abs(avg) > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            else:
                info += ' %s' % self.sum_values[k]

        self.total_width += len(info)
        if prev_total_width > self.total_width:
            info += ((prev_total_width - self.total_width) * ' ')

        sys.stdout.write(info)
        sys.stdout.flush()

        if current >= self.target:
            sys.stdout.write("\n")
            
    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)
        
    def erase(self):
        prev_total_width = self.total_width
        sys.stdout.write('\b' * prev_total_width)
        
        

"""class ProgressLogger(Callback):
    def __init__(self, mode='samples'):
        if(mode == 'samples'):
            self.use_steps = False
        elif(mode == 'steps'):
            self.use_steps = True
        else:
            raise ValueError('Unknown mode {}'.format(mode))

    def epoch_start(self, epoch, logs=None):
        if(self.use_steps):
            self.target = self.params['steps']
        else:
            self.target = self.params['samples']
        self.progbar = Progbar(target=self.target)
        self.seen = 0
        
    def epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            train_label = 'train_'+k
            valid_label = 'valid_'+k
            if(k in logs):
                self.log_values.append((k, logs[k]))
            if(train_label in logs):
                self.log_values.append((train_label, logs[train_label]))
            if(valid_label in logs):
                self.log_values.append((valid_label, logs[valid_label]))
        self.progbar.update(self.seen, self.log_values, force=True)
        self.progbar.erase()
                
        
    def batch_start(self, batch, logs=None):
        if(self.seen < self.target):
            self.log_values = []

    def batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if(self.use_steps):
            self.seen += 1
        else:
            self.seen += batch_size
        
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
                
        if(self.seen < self.target):
            self.progbar.update(self.seen, self.log_values)

    def train_start(self, logs=None):
        self.epochs = self.params['n_epochs']"""
