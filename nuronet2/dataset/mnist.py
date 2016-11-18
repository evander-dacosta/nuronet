# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 22:29:38 2016

@author: evander
"""

import scipy.io
import matplotlib.pyplot as plt
import seaborn
import numpy
import numpy.random

from utils.mnistLoad import load_data
from densedataset import DenseDataset
from nuronet2.backend import N


def mnistLoader(fName, limit=None):
    train, valid, test = load_data(fName, limit=limit)
    return (train[0], train[1], test[0], test[1])
    
class MnistDataset(DenseDataset):
    def __init__(self, fName, limit=None,
                 flatten=True, single_channel=False, batch_size=10,
                 validation=0.1, test=0.1):
        self.flatten = flatten
        self.singleChannel = single_channel

        X, Y, XTest, YTest = mnistLoader(fName, limit)
        Y = Y.astype('int32')
        YTest = YTest.astype('int32')
        if(not flatten):
            if(self.singleChannel):
                X = X.reshape(X.shape[0], 1, 28, 28)
                XTest = XTest.reshape(XTest.shape[0], 1, 28, 28)
            else:
                X = X.reshape(X.shape[0], 28, 28)
                XTest = XTest.reshape(XTest.shape[0], 28, 28)
        else:
            if(self.singleChannel):
                X = X.reshape(X.shape[0], 1, X.shape[-1])
                XTest = XTest.reshape(XTest.shape[0], 1, XTest.shape[-1])

        DenseDataset.__init__(self, X, Y, XTest, YTest,
                              batchSize = batch_size, validation=validation)

    def plot(self, image):
        if(isinstance(image, int)):
            image = self.X[image]
            if(self.singleChannel):
                image = image[0]
        if(self.flatten):
            image = image.reshape(28, 28)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        
    def predictTrain(self, yPred, image):
        self.plot(image)
        return numpy.argmax(yPred([self.X[image]]))
        
    def predict(self, yPred, image):
        image = self.XTest[image]
        self.plot(image)
        return numpy.argmax(yPred([image]))
        
    def accuracy(self, yPred):
        correct = 0
        for i in range(len(self.XTest)):
            ans = numpy.argmax(yPred([self.XTest[i]]))
            real = self.YTest[i]
            if(ans == real):
                correct += 1
        return correct / float(len(self.XTest))