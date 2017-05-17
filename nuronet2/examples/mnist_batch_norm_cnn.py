#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:17:00 2017

@author: evander
"""

    from nuronet2.base import *
    from nuronet2.dataset.mnist import MnistDataset
    from nuronet2.layers import *
    from nuronet2.optimisers import *
    from test import *


    folderName='/home/evander/Dropbox/data/mnist/mnist.pkl.gz'

    data = MnistDataset(folderName, limit=None, flatten=False, 
                        single_channel=True,
                        batch_size=64,
                        validation=0.01)
    
    add_batch_norm = True

    net = NeuralNetwork()
    
    net.add(Conv2dLayer((32, 3, 3), activation="linear", input_shape=(1, 28, 28)))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(Activation('relu'))
    net.add(Conv2dLayer((32, 3, 3), activation="linear"))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(Activation('relu'))
    net.add(Maxpool2d(pool_size=(2, 2), strides=(1, 1)))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))

    net.add(Conv2dLayer((64, 3, 3), activation="linear"))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(Activation('relu'))
    net.add(Conv2dLayer((64, 3, 3), activation="linear"))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(Activation('relu'))
    net.add(Maxpool2d(pool_size=(2, 2), strides=(1, 1)))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
        
        
    net.add(Conv2dLayer((128, 3, 3), activation="linear"))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(Activation('relu'))
    net.add(Conv2dLayer((128, 3, 3), activation="linear"))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(Activation('relu'))
    net.add(Maxpool2d(pool_size=(2, 2), strides=(1, 1)))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))

    net.add(Flatten())
    net.add(Dropout(0.5))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(DenseLayer(512, activation="linear"))
    if(add_batch_norm):
        net.add(BatchNormalisation(axis=1))
    net.add(Activation('relu'))
    net.add(Dropout(0.25))
    net.add(DenseLayer(10, activation="softmax"))
    
    
    net.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    history = net.fit_dataset(data, 10)
    
    def get_accuracy(x_t, y_t):
        y_t = numpy.argmax(y_t, axis=-1)
        pred = net.predict(x_t)
        pred = numpy.argmax(pred, axis=-1)
        assert(pred.shape == y_t.shape)
        return numpy.sum(pred == y_t)
        
    count = 0
    samples = 0
    for i in range(10):
        x_t = data.x_test[i*1000:(i+1)*1000]
        y_t = data.y_test[i*1000:(i+1)*1000]
        samples += x_t.shape[0]
        count += get_accuracy(x_t, y_t)
    print "Training accuracy is {}%".format(count*100./float(samples))