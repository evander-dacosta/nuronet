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
    data = IrisDataset(f_name=fName, batch_size=8, validation=0.1)
    
    model = NeuralNetwork()
    model.add(DenseLayer(100, w_regulariser={'name':'l2', 'l2':1e-4},
                        activation="tanh2", input_shape=(3,)))
    model.add(DenseLayer(3, activation="softmax"))
    
    model.compile('rmsprop', "categorical_crossentropy")
    h = model.fit_generator(data, n_epochs=20, n_workers=1)
    
    test = numpy.argmax(model.predict(data.x_test), axis=1)
    real = data.y_test.nonzero()[1]

    print test
    print real
    g = test - real
    non = g.nonzero()[0]
    print (1. - (len(non) / float(g.shape[0]))) * 100.