# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 00:07:10 2016

@author: evander
"""

import numpy
import matplotlib.pyplot as plt
from time import time
from tabulate import tabulate
from collections import OrderedDict
from backend import N
from mlmodel import MLModel
from objectives import get_objective


class TrainingLog:
    def __init__(self, optimiser):
        self.printHeaders = True
        self.optimiser = optimiser

    def __call__(self):
        print self.table(self.optimiser.train_history)

    def table(self, trainHistory):
        info = trainHistory[-1]
        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train loss', '{:.5f}'.format(info['trainLoss']))
        ])

        if('validLoss' in info.keys()):
            info_tabulate['validLoss'] = "{:.5f}".format(info['validLoss'])
        info_tabulate['duration'] = "{:.2f}s".format(info['duration'])

        tab = tabulate([info_tabulate], headers="keys", floatfmt='.5f')
        out = ""
        if(self.printHeaders):
            out = "\n".join(tab.split('\n', 2)[:2])
            out += "\n"
            self.printHeaders = False

        out += tab.rsplit("\n", 1)[-1]
        return out
        

class Optimiser(object):
    def __init__(self, model, objective):
        self._train_history = []
        self._current_epoch = 0
        self._i = N.scalar(dtype='int32', name='i')
        self.callbacks = [TrainingLog(self)]
        self._lr = N.shared(0.)
        
        self._model = None
        self._dataset = None
        self.nEpochs = 0
        
        self.set_model(model)
        ##########################################
        self._objective = get_objective(objective)
        ##########################################
        
    @property
    def learning_rate(self):
        return self._lr
        
    @learning_rate.setter
    def learning_rate(self, value):
        N.set_value(self._lr, value)
        
    @property
    def dataset(self):
        return self._dataset
        
    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        
    @property
    def train_history(self):
        return self._train_history
        
    @property
    def epoch(self):
        return self._current_epoch
        
    def add_to_history(self, epochInfo):
        self._train_history.append(epochInfo)
        
    def add_callback(self, callback):
        self.callbacks.append(callback)
        
    def reset_callbacks(self):
        trainingLog = self.callbacks[0]
        self.callbacks = [trainingLog]
        
    def end_of_epoch(self):
        [callback() for callback in self.callbacks]
        
    def increment_epoch(self):
        self._current_epoch = self._current_epoch + 1
        
    def reset(self):
        pass
    
    def plot(self, start=0, stop=None, figure=None, *args, **kwargs):
        """
        Graphs the stored per-epoch convergence of the
        model

        TODO: Make this realtime

        Parameters
        ----------
        figure : A predetermined figure to draw on
        """
        if(len(self._train_history) == 0):
            return
        if(figure is None):
            figure = plt.figure()

        stop = len(self._train_history) if stop is None else stop
        subplot = figure.add_subplot(111)
        trainLosses = []
        validLosses = []

        for i in xrange(start, stop):
            info = self._train_history[i]
            epoch = info['epoch']
            trainLosses.append([epoch, info['trainLoss']])
            if('validLoss' in info.keys()):
                validLosses.append([epoch, info['validLoss']])

        trainLosses = numpy.asarray(trainLosses)
        validLosses = numpy.asarray(validLosses)

        trainingPlot, = subplot.plot(trainLosses[:, 0], trainLosses[:, 1])
        if(len(validLosses) > 0):
            validPlot, = subplot.plot(validLosses[:, 0], validLosses[:, 1])
            figure.legend(
                [trainingPlot, validPlot], ['trainingError', 'validation error'])
        else:
            figure.legend([trainingPlot], ['training error'])

        figure.show()
        
    def set_num_epochs(self, value):
        self.nEpochs = value
        
        
    def set_model(self, model):
        if(not isinstance(model, MLModel)):
            raise Exception("Unknown model type")
        self._model = model
        
    def get_train_valid_funcs(self):
        if(self._model is None):
            raise Exception("Model not set for current optimiser")
        main_cost = self._objective(self._model.output, self._model.prop_up())
        side_costs = self._model.get_cost()
        total_cost = main_cost + side_costs
        
        params = self._model.get_params()
        updates = self.get_updates(total_cost, params).items()
        updates += self._model.get_updates()
        inputs = [self._model.input]
        if(self._model.supervised):
            inputs += [self._model.output]
        train_func = N.function(inputs, outputs=[total_cost],
                                updates=updates)
        valid_func = N.function(inputs, outputs=[total_cost])
        
        return train_func, valid_func
        
        
    def trainLoop(self, dataset):
        trainFunction, validFunction = self.get_train_valid_funcs()
        bestTrainLoss = numpy.inf
        bestValidLoss = numpy.inf

        for epoch in xrange(self.nEpochs):
            XTrain, YTrain, XValid, YValid = dataset.validation_split()
            trainLosses = []
            t0 = time()

            for Xb, Yb in dataset(XTrain, YTrain):
                trainLoss = trainFunction(
                    Xb) if not self._model.supervised else trainFunction(Xb, Yb)
                trainLosses += trainLoss

            validLoss = validFunction(
                    XValid) if not self._model.supervised else validFunction(XValid, YValid)

            t1 = time()

            averageTrainLoss = numpy.mean(trainLosses)
            averageValidLoss = numpy.mean(validLoss)
            if(averageTrainLoss < bestTrainLoss):
                bestTrainLoss = averageTrainLoss

            epochInfo = {
                'epoch': epoch + 1,
                'trainLoss': averageTrainLoss,
                'bestTrainLoss': bestTrainLoss,
                'validLoss': averageValidLoss,
                'bestValidLoss': bestValidLoss,
                'duration': t1 - t0
            }

            self.add_to_history(epochInfo)
            self.increment_epoch()
            self.end_of_epoch()
            
    def fit(self, dataset, nEpochs):
        if(not self.model.is_built):
            self.model.build()
        self.dataset = dataset
        self.set_model_input_output(self.model, self.dataset)
        self.reset()
        self.set_num_epochs(nEpochs)
        self.trainLoop(dataset)
        
    def set_model_input_output(self, model, dataset):
        """
        Sets the models input and output to match
        the dataset's input and output
        """
        model.input = N.variable(ndim=dataset.X.ndim,
                                       dtype=dataset.X.dtype)
        if(hasattr(dataset, 'Y')):
            output_dtype = dataset.Y.dtype
        else:
            output_dtype = 'float32'
        model.output = N.variable(ndim=len(model.output_shape),
                                        dtype=output_dtype)

    def get_updates(self, cost, params):
        """
        TODO:DOC
        """
        raise NotImplementedError()