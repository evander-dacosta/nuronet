# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:26:31 2016

@author: Evander
"""

import numpy
from backend import N
from densedataset import DenseDataset


class Iris(object):

    @staticmethod
    def getFlowerIndex(stringArray):
        """
        Returns a numerical index 0 - 2 depending on
        the name of the flower in stringArray.
        0 : Iris Setosa
        1 : Iris Versicolour
        2 : Iris Virginica
        """
        arrayLength = len(stringArray)
        versicolor = numpy.array(['Iris-versicolor'] * arrayLength)
        virginica = ['Iris-virginica'] * arrayLength

        isVersicolor = numpy.core.defchararray.equal(
            stringArray,
            versicolor).astype('int32')
        isVirginica = numpy.core.defchararray.equal(
            stringArray,
            virginica).astype('int32')

        indices = isVersicolor + (2 * isVirginica)
        y = numpy.zeros((arrayLength, 3))
        y[numpy.arange(arrayLength), indices] = 1
        return y

    @staticmethod
    def readFile(fName, dtype):
        try:
            f = open(fName)
        except:
            raise Exception("Could not open file %s" % fName)

        dataPoints = numpy.array([line[:-1].split(',') for line in
                                  f.readlines()[:-1]])
        numpy.random.shuffle(dataPoints)
        X = dataPoints[:, :3].astype(dtype)
        Y = dataPoints[:, 4]
        Y = Iris.getFlowerIndex(Y).astype(dtype)
        
        trainLength = int(0.9 * X.shape[0])
        return X[:trainLength], Y[:trainLength], X[trainLength:], \
                                    Y[trainLength:]


class IrisDataset(DenseDataset):

    def __init__(self, batchSize=1, fName="/home/evander/Desktop/data/iris/iris.data",
                 validation=0., shuffle=False, supervised=True):
        X, Y, XTest, YTest = Iris.readFile(fName, dtype=N.default_dtype)
        DenseDataset.__init__(self, X, Y, XTest, YTest, batchSize=batchSize,
                              validation=validation, shuffle=shuffle,
                              supervised=supervised)
        
if __name__ == "__main__":
    iris = IrisDataset(batchSize=5, validation=0.1)
    x, y, xValid, yValid = iris.validation_split()
        
