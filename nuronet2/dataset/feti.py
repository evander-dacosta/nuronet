# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:27:30 2016

@author: Evander
"""

import scipy.io
import matplotlib.pyplot as plt
import seaborn
import numpy
import numpy.random

from densedataset import DenseDataset
from backend import N

def window(array, width, stepSize=1):
    assert(len(array.shape) == 1)
    array = array.reshape((array.shape[0], 1))
    n = array.shape[0]
    return numpy.hstack(array[i: 1+n+i-width : stepSize] for i in range(0, width))
    
    
def preprocess(array, eps = 1e-2):
    #scale between [-1, 1]
    maxVal = array.max()
    minVal = array.min()
    if(numpy.abs(maxVal) > numpy.abs(minVal)):
        scalar = maxVal
    else:
        scalar = minVal
    return array / (scalar + eps)


class AliciaSetLoader(object):

    def __init__(self, fName, seqLength=100, animal = None):
        """
        The shape of self.x, and self.y is (numSequences, sequenceLength, 1)
        """
        if(animal is None):
            yLabel = 'y'
        else:
            yLabel = 'An{}y'.format(animal)
        data = scipy.io.loadmat(fName)
        self.seqLength = seqLength
        self.x = data['x'][:, 0].astype(N._default_dtype)
        self.y = data[yLabel][:, 0].astype(N._default_dtype)
        self.t = data['t'][0, :].astype(N._default_dtype)

    def shape(self, arr):
        assert(arr.shape[0] % self.seqLength == 0)
        arr = arr.reshape((arr.shape[0] / self.seqLength, self.seqLength,  1))
        return arr
        
    def shapeWindow(self, arr, stepSize):
        data = window(arr, self.seqLength, stepSize)
        return data.reshape((data.shape[0], data.shape[1], 1))

    def plotSection(self, length=None, prediction=None):
        assert(isinstance(length, (list, tuple, int, type(None))))
        if(isinstance(length, type(None))):
            length = [0, self.t.shape[0]]
        if(isinstance(length, int)):
            length = [0, length]
        plotT = self.t[length[0]: length[1]]
        plotX = self.x[length[0]: length[1]]
        #plotY = self.y[length[0] : length[1]]

        f, s = plt.subplots(2, 1)
        s[0].plot(plotT, plotX)
        if(prediction is not None):
            s[1].plot(prediction[0], prediction[1])
        plt.show()
            

class FetiDataset(DenseDataset):

    def __init__(
            self,
            chunkSize=100,
            batchSize = 1,
            fName="/home/evander/Desktop/data/aliciaData/50HzGWN.mat",
            validation=0., windowStepSize=None, test=0.1):
        """
        Initialises the FETI Dataset. Uses Feti average

        Parameters
        ----------
        chunkSize : Size of each temporal 'chunk'
        validation: Proportion of dataset to be used in cross-validation
        """
        self.dataset = AliciaSetLoader(fName, chunkSize)
        self.isWindowed = False
        if(isinstance(windowStepSize, (int, bool))):
            self.isWindowed = True
            if(isinstance(windowStepSize, bool) and windowStepSize):
                windowStepSize = 1                   
            if(isinstance(windowStepSize, int)):
                assert(windowStepSize > 0)
        
        x = preprocess(self.dataset.x)
        y = preprocess(self.dataset.y)
                
        x, y, xTest, yTest = self.getTestTrainData(x, y, test)
                                       
        if(self.isWindowed):
            x = self.dataset.shapeWindow(x, windowStepSize)
            y = self.dataset.shapeWindow(y, windowStepSize)
            
        else:
            x = self.dataset.shape(x)
            y = self.dataset.shape(y)
            
        
        self.t = self.dataset.t
        DenseDataset.__init__(self, X=x,
                              Y=y, XTest = xTest, 
                              YTest=yTest, batchSize=batchSize,
                              validation=validation)
                              
    def getTestTrainData(self, x, y, testRatio):
        indx = int(testRatio * x.shape[0])
        delim = numpy.random.randint(0, x.shape[0] - indx)
        range = numpy.arange(delim, delim + indx)
        xTest = x[range]
        yTest = y[range]
        x = numpy.delete(x, range)
        y = numpy.delete(y, range)
        return x, y, xTest, yTest
                              
    
                              
    def plotPredictor(self, predictor):
        x = self.XTest.reshape((1, self.XTest.shape[0], 1))
        y = self.YTest.reshape((self.YTest.shape[0], 1))
        yPred = predictor(x)
        plt.plot(y)
        plt.plot(yPred.reshape((numpy.prod(yPred.shape),)))
        
    def mse(self, predictor, start=0, stop=None):
        x = self.XTest[start:stop]
        y = self.YTest[start:stop]
        x = x.reshape((1, x.shape[0], 1))
        y = y.reshape((y.shape[0], 1))
        yP = predictor(x)
        dif = numpy.sum(numpy.square(y - yP))
        denom = numpy.sum(numpy.square(y))
        return dif * 100./denom
            
        
        
        
        
        
        
        
class FetiAnimalDataset(FetiDataset):
    def __init__(
            self,
            animal = 1,
            chunkSize=100,
            batchSize = 1,
            fName="/home/evander/Desktop/data/aliciaData/50HzGWN_Individuals.mat",
            validation=0.,
            test = 0.1, windowStepSize=None):
        """
        Initialises the FETI Dataset. Uses animal specified by 'animal'

        Parameters
        ----------
        chunkSize : Size of each temporal 'chunk'
        validation: Proportion of dataset to be used in cross-validation
        animal: Which animal to sample from. 1 - 5
        """
        assert(animal in range(1, 6)+[None])
        self.dataset = AliciaSetLoader(fName, seqLength = chunkSize,
                                       animal = animal)
        self.isWindowed = False
        if(isinstance(windowStepSize, (int, bool))):
            self.isWindowed = True
            if(isinstance(windowStepSize, bool) and windowStepSize):
                windowStepSize = 1                   
            if(isinstance(windowStepSize, int)):
                assert(windowStepSize > 0)
                
        x = preprocess(self.dataset.x)
        y = preprocess(self.dataset.y)
                
        x, y, xTest, yTest = self.getTestTrainData(x, y, test)
                                       
        if(self.isWindowed):
            x = self.dataset.shapeWindow(x, windowStepSize)
            y = self.dataset.shapeWindow(y, windowStepSize)
            
        else:
            x = self.dataset.shape(x)
            y = self.dataset.shape(y)            

            
        self.t = self.dataset.t
        DenseDataset.__init__(self, X=x,
                              Y=y, XTest=xTest,
                              YTest=yTest, batchSize=batchSize,
                              validation=validation)
        
if __name__ == "__main__":
    feti = FetiAnimalDataset(batchSize = 15, validation = 0.2, windowStepSize=1)
    a, b, c, d = feti.validationSplit()
