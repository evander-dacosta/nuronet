# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:40:50 2016

@author: evander
"""

import numpy
import cPickle
import matplotlib.pyplot as plt
from densedataset import DenseDataset

labels = ['aeroplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
              'frog', 'horse', 'ship', 'truck']

def processCifar(imageSet, flatten):
    if(not flatten):
        imageSet = imageSet.reshape((imageSet.shape[0], 3, 32, 32))
    imageSet = imageSet / 255.
    return imageSet
    
def plotCifar(image):
    image = numpy.swapaxes(image, 1, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image.T)
    
def labelCifar(num):
    assert(num < 10)
    return labels[num]

def loadCifar(folder, limit = 5, flatten = False):
    limit -= 1
    assert(limit >= 0 and limit < 5)
    files = ['/data_batch_{}'.format(item) for item in xrange(1, 6)]
    testFile = '/test_batch'
    loadableFiles = [files[i] for i in xrange(limit + 1)]

        
    X = numpy.zeros((10000 * len(loadableFiles), 3, 32, 32))
    Y = numpy.zeros((10000 * len(loadableFiles), 10))
    labels = []
    for i, f in enumerate(loadableFiles):
        fName = folder + f
        with open(fName, 'rb') as fi:
            dataNew = cPickle.load(fi)
            X[i * 10000 : (i + 1) * 10000] = processCifar(dataNew['data'], flatten)
            labels += dataNew['labels']
    Y[numpy.arange(Y.shape[0]), labels] = 1
    
    #load test
    
    fName = folder + testFile     
    with open(fName, 'rb') as fi:
        dataN = cPickle.load(fi)
        XTest = processCifar(dataN['data'], flatten)
        test_labels = dataN['labels']
    YTest = numpy.zeros((XTest.shape[0], 10))
    YTest[numpy.arange(YTest.shape[0]), test_labels] = 1
    return numpy.array(X, 'float32'), numpy.array(Y, 'float32'), \
            numpy.array(XTest, 'float32'), numpy.array(YTest, 'float32')


class Cifar10Dataset(DenseDataset):
    def  __init__(self, folderName,
                   limit = 5, flatten = False, validation = 0.,
                   batch_size = 10):
        self.flatten = flatten
        X, Y, XTest, YTest = loadCifar(folderName, limit=limit, flatten=flatten)
        DenseDataset.__init__(self, X, Y, XTest, YTest, batch_size=batch_size,
                              validation=validation)
        
    def plot(self, image):
        plotCifar(self.XTest[image])
        
    def label(self, index):
        return labels[index]
        
    def accuracy(self, yPred, limit=None):
        a = numpy.argmax(yPred(self.XTest[:limit]), axis=1)
        labels = self.YTest[:limit].nonzero()[1]
        correct = a - labels
        return (len(a) - len(correct.nonzero()[0]))/float(len(a))
        
    def predict(self, yPred, number):
        print "Prediction: {}".format(self.label(numpy.argmax(yPred(self.XTest[number:number+1]))))
        print "Actual: {}".format(self.label(numpy.argmax(self.YTest[number])))
        self.plot(number)
        plt.show()
        return 