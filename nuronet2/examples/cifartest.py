# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:19:04 2016

@author: evander
"""

from nuronet2.base import *
from nuronet2.dataset.cifar10 import loadCifar, Cifar10Dataset
from nuronet2.layers import *
from nuronet2.optimisers import *
from test import *


"""
Because cifar files are so large,
we use net.fit_generator() to train
"""
        
if __name__ == "__main__":
    folderName='/home/evander/Dropbox/data/cifar-10'
    X, Y, XTest, YTest = loadCifar(folderName, limit=5, flatten=False)
    
    data = TestIterator(X, Y, batch_size=32, shuffle=True, validation=0.001)

    net = NeuralNetwork((3, 32, 32))
    
    net.add(Conv2dLayer((32, 3, 3), activation="relu"))
    net.add(Conv2dLayer((32, 3, 3), activation="relu"))
    net.add(Maxpool2d(pool_size=(2, 2), strides=(1, 1)))
    net.add(Dropout(0.25))
    
    net.add(Conv2dLayer((64, 3, 3), activation="relu"))
    net.add(Conv2dLayer((64, 3, 3), activation="relu"))
    net.add(Maxpool2d(pool_size=(2, 2), strides=(1, 1)))
    net.add(Dropout(0.25))
    
    net.add(Flatten())
    net.add(DenseLayer(512, activation="relu"))
    net.add(Dropout(0.5))
    net.add(DenseLayer(10, activation="softmax"))
    
    
    net.compile('adam', 'categorical_crossentropy')
    history = net.fit_generator(data, 20)
    
    """net.load_weights("cifarNet")
    yPred = net.predict
    print data.accuracy(yPred, limit=1000)"""