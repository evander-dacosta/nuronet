# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:19:04 2016

@author: evander
"""

from nuronet2.base import *
from nuronet2.dataset.cifar10 import Cifar10Dataset
from nuronet2.layers import *
from nuronet2.optimisers import *
from test import *



        
if __name__ == "__main__":
    folderName='/home/evander/Dropbox/data/cifar-10'
    data = Cifar10Dataset(folderName=folderName, limit=3,
                          batchSize=32, validation=0.1)
    
    
    inp = Input((3, 32, 32))
    layer1 = Conv2dLayer((32, 3, 3), activation="relu")(inp)
    layer2 = Conv2dLayer((32, 3, 3), activation="relu")(layer1)
    pool = Maxpool2d(pool_size=(2, 2), strides=(1, 1))(layer2)
    dpOut = Dropout(0.25)(pool)
    
    layer3 = Conv2dLayer((64, 3, 3), activation="relu")(dpOut)
    layer4 = Conv2dLayer((64, 3, 3), activation="relu")(layer3)
    pool2 = Maxpool2d(pool_size=(2, 2), strides=(1, 1))(layer4)
    dp2 = Dropout(0.25)(pool2)
    
    flat = Flatten()(dp2)    
    dense1 = DenseLayer(512, activation="relu")(flat)
    dpOut1 = Dropout(0.5)(dense1)
    out = DenseLayer(10, activation="softmax")(dpOut1)
    
    
    model = NetworkModel(inp, out)
    
    optim = Adadelta(model, 'categorical_crossentropy')
    optim.fit(data, 20)
    
    #model.load_weights("cifar.nnet")
    #yPred = model.get_predictor()
    #print data.accuracy(yPred, limit=1000)