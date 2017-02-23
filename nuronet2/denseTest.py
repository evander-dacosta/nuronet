# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:10:56 2016

@author: evander
"""

import numpy
from nuronet2.layers import *
from nuronet2.optimisers import *
from nuronet2.dataset.iris import IrisDataset
from nuronet2.backend import N


from nuronet2.base import MLModel, NetworkModel
        


if __name__ == "__main__":
    #fName = "/home/evander/Dropbox/data/iris/iris.data"
    fName = "C:\\Users\\Evander\\Dropbox\\data\\iris\\iris.data"
    data = IrisDataset(fName=fName, validation=0.1)
    
    layer1 = Input((3,))
    layer2 = DenseLayer(100, w_regulariser={'name':'l2', 'l2':1e-4},
                        activation="tanh2")
    layer3 = DenseLayer(3, activation="softmax")
    
    out = layer3(layer2(layer1))
    
    model = NetworkModel(layer1, out)
    
    optim = Adam(model, ["categorical_crossentropy"])
    optim.fit(data, 20)
    
    test = numpy.argmax(model.predict(data.XTest), axis=1)
    real = data.YTest.nonzero()[1]
    print test
    print real
    g = test - real
    non = g.nonzero()[0]
    print (1. - (len(non) / float(g.shape[0]))) * 100.
    