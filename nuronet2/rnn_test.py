# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 22:22:42 2017

@author: Evander
"""

from nuronet2.optimisers import *
from nuronet2.dataset.iris import *
from nuronet2.backend import N
import numpy
import scipy.io

from nuronet2.base import MLModel, NetworkModel, NeuralNetwork
from nuronet2.layers import DenseLayer, RNNLayer


class SpikeLoader:
    def __init__(self, fName, seqLength=100):
        data = scipy.io.loadmat(fName)
        x = data['x'][0]
        y = data['y'][0]
        self.x, self.y = self.normalise(x, y)
        self.seqLength = seqLength
        
    def normalise(self, x, y):
        self.max = numpy.abs(numpy.max(x))
        self.mean = numpy.mean(x)
        x = x / numpy.abs(numpy.max(x))
        #y = y / numpy.abs(numpy.max(y))
        return x - numpy.mean(x), y
        
    def shape(self, arr, clip=None):
        if(clip is not None):
            arr = arr[:-clip]
        if(arr.shape[0] % self.seqLength != 0):
            raise Exception("Wrong shape: {}. Must be divisible by {} (set argument 'clip' for this function)".format(arr.shape[0],
                            self.seqLength))
        arr = arr.reshape((arr.shape[0] / self.seqLength, self.seqLength,  1))
        return arr
        
if __name__ == "__main__":
    fName = "/home/evander/Dropbox/data/spikingData/1008.mat"
    #fName = "C:\\Users\\Evander\\Dropbox\\data\\spikingData\\1008.mat"
    spikeLoader = SpikeLoader(fName)
    chunk_size = 200
    data_size = spikeLoader.x.shape[0] / chunk_size
    x = spikeLoader.x.reshape((data_size, chunk_size, 1))
    y = spikeLoader.y.reshape((data_size, chunk_size, 1))
    
    model = NeuralNetwork()
    model.add(RNNLayer(64, input_shape=(None, 1)))
    model.add(DenseLayer(1, activation="sigmoid"))
    
    model.compile('adam', 'binary_crossentropy')
    h = model.fit(x, y, batch_size=8, validation_split=0.1, n_epochs=400)
    
    