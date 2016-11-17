# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:17:58 2016

@author: Evander
"""

import numpy
import itertools
from sklearn.cross_validation import KFold, StratifiedKFold
from dataset import Dataset, _sldict

class DenseDataset(Dataset):

    """An abstract class representing a dataset
       which can be used to train various ML models
       efficiently.

       Parameters
       ----------
       X : Input Data
       Y : Input Targets (optional)
       (X must be Theano shared variable if shared = True)
       batchSize : Desired output batchsize (optional)
    """

    def __init__(self, X, Y=None, XTest = None, YTest = None,
                 batchSize=None, validation=0.,
                 shuffle=True, supervised=True):
        assert(isinstance(batchSize, int) and batchSize is not None)
        assert(isinstance(validation, (int, float)))
        Dataset.__init__(self, batchSize)
        self.X = X
        self.Y = Y
        self.XTest = XTest
        self.YTest = YTest
        self.shuffle = shuffle

        self.supervised = supervised
        if(self.Y is None):
            self.supervised = False
            
        #if unsupervised, we want the test targets same as the test inputs
        if(not self.supervised):
            self.Y = self.X
            self.YTest = self.XTest

        if(shuffle):
            self.shuffler()
        
        # get the Y's dtype to find out whether it's a
        # label or regression problem
        self.regression = self.is_regression_problem()

        self.validProportion = validation
        if(self.validProportion > 0.5):
            raise Exception("Validation proportion is too high." + \
                            " Recommend less than 0.5")
        
        self.iterator = None
        
    def shuffler(self):
        xIndices = numpy.arange(0, self.X.shape[0], 1)
        numpy.random.shuffle(xIndices)
        self.X = self.X[xIndices]
        self.Y = self.Y[xIndices]

    def set_supervised(self, value):
        assert(isinstance(value, bool))
        self.supervised = value

    def is_regression_problem(self):
        if(self.Y is not None):
            if(len(self.Y.shape) > 1 or str(self.Y.dtype).startswith('float')):
                return True
        return False
    
    def iteration_split(self):
        if(self.validProportion > 0.):
            nFolds = round(1. / self.validProportion)
            if(self.regression):
                kf = KFold(self.Y.shape[0], nFolds)
            else:
                kf = StratifiedKFold(self.Y, nFolds)
        else:
            def ret():
                yield (numpy.arange(self.X.shape[0]), [])
            kf = ret()
        return itertools.cycle(kf)

    def validation_split(self):
        if(self.iterator is None):
            self.iterator = self.iteration_split()
        trainIndices, validIndices = next(self.iterator)
        XTrain = self.X[trainIndices]
        YTrain = self.Y[trainIndices]
        XValid = self.X[validIndices]
        YValid = self.Y[validIndices]
        return XTrain, YTrain, XValid, YValid
