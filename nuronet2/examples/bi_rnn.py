
from nuronet2.optimisers import *
from nuronet2.dataset.mnist import *
from nuronet2.backend import N
import numpy
import scipy.io

from nuronet2.base import MLModel, NetworkModel, NeuralNetwork, Input
from nuronet2.layers import DenseLayer, RNNLayer, Merge


if __name__ == "__main__":
    data = MnistDataset(fName="/home/evander/Dropbox/data/mnist/mnist.pkl.gz",
                        batch_size=32, flatten=False, limit=None)
    n_epochs = 30

    input = Input(shape=(28, 28))
    forward_rnn = NeuralNetwork()
    forward_rnn.add(RNNLayer(32, return_sequences=False, input_shape=(28, 28)))
    
    backward_rnn = NeuralNetwork()
    backward_rnn.add(RNNLayer(32, return_sequences=False, input_shape=(28, 28),
                       go_backwards=True))
    
    
    concat = Merge()([model(input), model_two(input)])
    output = DenseLayer(10, activation="softmax")(concat)
    
    bi_rnn = NetworkModel(inputs=input, outputs=output)    
    bi_rnn.compile("adam", "categorical_crossentropy")    
    bi_rnn.fit_dataset(data, n_epochs=n_epochs)
    
    y_ = bi_rnn.predict(data.x_test)
    assess = (numpy.argmax(y_, axis=1) == numpy.argmax(data.y_test, axis=1))
    print "Accuracy = {}".format(numpy.sum(assess) / float(assess.shape[0]))