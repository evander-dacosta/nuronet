from nuronet2.optimisers import *
from nuronet2.dataset.mnist import *
from nuronet2.backend import N
import numpy
import scipy.io

from nuronet2.base import MLModel, NetworkModel, NeuralNetwork, Input
from nuronet2.layers import DenseLayer, RNNLayer, AddMerge, Layer
from nuronet2.layers.layer import Dropout, Activation
from nuronet2.activations import get_activation



if __name__ == "__main__":
    from nuronet2.dataset.iris import *
    #fName = "/home/evander/Dropbox/data/iris/iris.data"
    fName = "C:\\Users\\Evander\\Dropbox\\data\\iris\\iris.data"
    X, Y, XTest, YTest = Iris.readFile(fName, dtype=N.default_dtype)
    
    input = Input((3,))
    h1 = DenseLayer(100, w_regulariser={'name':'l2', 'l2':1e-4},
                        activation="linear")(input)
                        
    h2 = DenseLayer(100, w_regulariser={'name':'l2', 'l2':1e-4},
                        activation="linear")(input)
    merge = AddMerge()([h1, h2])
    a = Activation('tanh')(merge)
    b = Dropout(0.1)(a)
    y = DenseLayer(3, activation="softmax")(b)
    
    
    model = NetworkModel(input, y)
    model.compile('adam', "categorical_crossentropy")
    h = model.fit(X, Y, batch_size=8, n_epochs=20)
    
    test = numpy.argmax(model.predict(XTest), axis=1)
    real = YTest.nonzero()[1]

    print test
    print real
    g = test - real
    non = g.nonzero()[0]
    print (1. - (len(non) / float(g.shape[0]))) * 100.
    