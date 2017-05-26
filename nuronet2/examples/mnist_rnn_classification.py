from nuronet2.optimisers import *
from nuronet2.dataset.mnist import *
from nuronet2.backend import N
import numpy
import scipy.io

from nuronet2.base import MLModel, NetworkModel, NeuralNetwork
from nuronet2.layers import DenseLayer, Activation, Dropout, RNNLayer, Conv1dLayer, Permute, BatchNormalisation



"""
    1) Baseline: 99% accuracy
    2) Training: 99.37% 
    3) Validation: 98.5%
    4) Test: 98.5%

    Can increase the models performance using data augmentation.
    Also, a more efficient gradient descent method that shrinks gradient sizes
    with epochs. 
"""

if __name__ == "__main__":
    fName="/home/evander/Dropbox/data/mnist/mnist.pkl.gz"
    #fName="C:\\Users\\Evander\\Dropbox\\data\\mnist\\mnist.pkl.gz"
    data = MnistDataset(fName=fName,
                        batch_size=32, flatten=False, limit=None)
    p = 0.6
    n_epochs = 500
    model = NeuralNetwork()
    model.add(Conv1dLayer((16, 10), strides=2, activation='linear', input_shape=(28, 28)))
    model.add(BatchNormalisation(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1dLayer((32, 5), strides=1, activation='linear'))
    model.add(BatchNormalisation(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1dLayer((64, 3), activation='linear'))
    model.add(BatchNormalisation(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1dLayer((128, 2), strides=1,activation='linear'))
    model.add(BatchNormalisation(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1dLayer((256, 2), strides=1, activation='linear'))
    model.add(BatchNormalisation(axis=1))
    model.add(Activation('relu'))
    model.add(Conv1dLayer((512, 2), strides=1,activation='linear'))
    model.add(BatchNormalisation(axis=1))
    model.add(Activation('relu'))
    model.add(Permute((2, 1)))
    model.add(Dropout(p))
    model.add(DenseLayer(128, activation='linear'))
    model.add(BatchNormalisation(axis=-1))
    model.add(Activation('tanh'))
    model.add(Dropout(p))
    model.add(DenseLayer(64, activation='linear'))
    model.add(BatchNormalisation(axis=-1))
    model.add(Activation('tanh'))
    model.add(Dropout(p))
    model.add(DenseLayer(32, activation='linear'))
    model.add(BatchNormalisation(axis=-1))
    model.add(Activation('tanh'))
    model.add(RNNLayer(8, return_sequences=False,
                       go_backwards=True, w_dropout=p / 2.))

    model.add(DenseLayer(10, activation="softmax"))
    
    model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
    
    model.fit_dataset(data, n_epochs=n_epochs)
    
    y_ = model.predict(data.x_test)
    assess = (numpy.argmax(y_, axis=1) == numpy.argmax(data.y_test, axis=1))
    print "Accuracy = {}".format(numpy.sum(assess) / float(assess.shape[0]))
    