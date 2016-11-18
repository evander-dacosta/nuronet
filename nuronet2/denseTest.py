# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:10:56 2016

@author: evander
"""

import numpy
from nuronet2.layers import *
from nuronet2.optimisers import *
from nuronet2.dataset.iris import IrisDataset


from nuronet2.base import MLModel
from test import AcyclicModel
        

if __name__ == "__main__":
    fName = "/home/evander/Dropbox/data/iris/iris.data"
    data = IrisDataset(fName=fName, validation=0.1)
    layer1 = Input((3,))
    layer2 = DenseLayer(50, w_regulariser={'name':'l2', 'l2':1e-4})
    layer3 = DenseLayer(3, activation="softmax")
    
    out = layer3(layer2(layer1))
    
    model = AcyclicModel(layer1, out)
    
    optim = Adam(model, ["categorical_crossentropy"])
    optim.fit(data, 20)
    print numpy.argmax(n.predict(data.XTest), axis=1)
    print data.YTest.nonzero()[1]