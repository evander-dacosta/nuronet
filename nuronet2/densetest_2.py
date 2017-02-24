# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:14:59 2017

@author: Evander
"""

import numpy
from nuronet2.layers import *
from nuronet2.optimisers import *
from nuronet2.dataset.iris import *
from nuronet2.backend import N

from nuronet2.base import MLModel, NetworkModel
from test import NeuralNetwork

if __name__ == "__main__":
    #fName = "/home/evander/Dropbox/data/iris/iris.data"
    fName = "C:\\Users\\Evander\\Dropbox\\data\\iris\\iris.data"
    X, Y, XTest, YTest = Iris.readFile(fName, dtype=N.default_dtype)
    
    layer1 = InputLayer((3,))
    layer2 = DenseLayer(100, w_regulariser={'name':'l2', 'l2':1e-4},
                        activation="tanh2")
    layer3 = DenseLayer(3, activation="softmax")
    """out = layer3(layer2(layer1))
    
    model = NetworkModel(layer1, out)"""
    model = NeuralNetwork()
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.compile('adam', "categorical_crossentropy")
    model.fit(X, Y, batch_size=2, n_epochs=20)
    
    test = numpy.argmax(model.predict(XTest), axis=1)
    real = YTest.nonzero()[1]
    print test
    print real
    g = test - real
    non = g.nonzero()[0]
    print (1. - (len(non) / float(g.shape[0]))) * 100.