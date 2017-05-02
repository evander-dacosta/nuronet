import numpy
from nuronet2.backend import N
from nuronet2.base import Layer, InputDetail, get_weightfactory, get_regulariser
from nuronet2.activations import get_activation


from nuronet2.layers import Recurrent


        
from nuronet2.optimisers import *
from nuronet2.dataset.mnist import *
from nuronet2.backend import N
import numpy
import scipy.io

from nuronet2.base import MLModel, NetworkModel, NeuralNetwork
from nuronet2.layers import DenseLayer, RNNLayer

if __name__ == "__main__":
    data = MnistDataset(fName="/home/evander/Dropbox/data/mnist/mnist.pkl.gz",
                        batch_size=32, flatten=False, limit=None)
    n_epochs = 30
    model = NeuralNetwork()
    model.add(LSTMLayer(64, return_sequences=False, input_shape=(28, 28),
                       go_backwards=True))
    model.add(DenseLayer(10, activation="softmax"))
    
    model.compile("adam", "categorical_crossentropy")
    
    model.fit_dataset(data, n_epochs=n_epochs)
    
    y_ = model.predict(data.x_test)
    assess = (numpy.argmax(y_, axis=1) == numpy.argmax(data.y_test, axis=1))
    print "Accuracy = {}".format(numpy.sum(assess) / float(assess.shape[0]))