
import numpy
import matplotlib.pyplot as plt
from time import time
from tabulate import tabulate
from collections import OrderedDict
from nuronet2.backend import N
from nuronet2.base import MLModel
from nuronet2.objectives import get_objective


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
    def __init__(self, model, objectives):
        self._train_history = []
        self._current_epoch = 0
        self._i = N.scalar(dtype='int32', name='i')
        self.callbacks = [TrainingLog(self)]
        self._lr = N.shared(0.)
        
        self._model = None
        self._dataset = None
        self.nEpochs = 0
        
        #set the model and the loss functions
        self.set_model(model)
        self.set_objectives(objectives)
        
    @property
    def model(self):
        return self._model
        
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
        if(not self._model.is_built):
            self._model.build()
        

    def set_objectives(self, objectives):
        """Can set multiple objectives with a list of objectives
        corresponding to the list of output tensors
        """
        if(isinstance(objectives, list)):
            if(len(objectives) != len(self.model.outputs)):
                raise Exception("When passing a list of objective functions, " + \
                                "it should have one entry per output tensor of " + \
                                "the model. The current model has {} tensors ".format(len(self.model.outputs)) + \
                                "whereas the number of objectives passed is {}.".format(len(objectives)))
            objective_functions = [get_objective(objective) for objective in objectives]
            
        else:
            obj_fun = get_objective(objectives)
            objective_functions = [obj_fun for _ in range(len(self.model.outputs))]
        self.objective_fuctions = objective_functions

    def compile(self):
        self.targets = []
        #prepare targets
        for tensor in self.model.outputs:
            ndim = len(tensor._nuro_shape)
            name = tensor.name
            dtype = N.dtype(tensor)
            self.targets.append(N.variable(ndim=ndim, dtype=dtype))
        
        #compute total loss
        total_loss = None
        for i in range(len(self.model.outputs)):
            y_target = self.targets[i]
            y_pred = self.model.outputs[i]
            if(total_loss is None):
                total_loss = self.objective_fuctions[i](y_target, y_pred)
            else:
                total_loss += self.objective_fuctions[i](y_target, y_pred)
        #add individual layer losses like regularisers
        total_loss += self.model.get_cost()
        self.total_loss = total_loss
    #TBI

        

    def get_train_valid_funcs(self):
        inputs = self.model.inputs + self.targets
        
        updates = self.get_updates(self.total_loss, self.model.trainable_weights).items()
        updates += self.model.get_updates().items()
        
        train_function = N.function(inputs, [self.total_loss], updates=updates)
        valid_function = N.function(inputs, [self.total_loss], updates=updates)
        return train_function, valid_function
        
    def fit(self, dataset, n_epochs):
        self.dataset = dataset
        self.reset()
        self.set_num_epochs(n_epochs)
        self.compile()
        
        train_function, valid_function = self.get_train_valid_funcs()
        bestTrainLoss = numpy.inf
        bestValidLoss = numpy.inf

        for epoch in xrange(self.nEpochs):
            XTrain, YTrain, XValid, YValid = dataset.validation_split()
            trainLosses = []
            t0 = time()

            for Xb, Yb in dataset(XTrain, YTrain):
                trainLoss = train_function(Xb, Yb)
                trainLosses += trainLoss

            validLoss = valid_function(XValid, YValid)

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
        
    def get_updates(self, cost, params):
        raise NotImplementedError()