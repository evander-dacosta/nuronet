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

from nuronet2.base import MLModel, NetworkModel, NeuralNetwork

if __name__ == "__main__":
    #fName = "/home/evander/Dropbox/data/iris/iris.data"
    fName = "C:\\Users\\Evander\\Dropbox\\data\\iris\\iris.data"
    X, Y, XTest, YTest = Iris.readFile(fName, dtype=N.default_dtype)
    
    model = NeuralNetwork()
    model.add(DenseLayer(100, w_regulariser={'name':'l2', 'l2':1e-4},
                        activation="tanh2",  input_shape=(3,)))
    model.add(DenseLayer(3, activation="softmax"))
    model.compile('adam', "categorical_crossentropy")
    h = model.fit(X, Y, batch_size=8, n_epochs=20, validation=0.)
    
    test = numpy.argmax(model.predict(XTest), axis=1)
    real = YTest.nonzero()[1]

    print test
    print real
    g = test - real
    non = g.nonzero()[0]
    print (1. - (len(non) / float(g.shape[0]))) * 100.