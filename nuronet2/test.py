import nuronet2 as nuro
from nuronet2.layers import Recurrent
from nuronet2.base import get_weightfactory
from nuronet2.activations import get_activation
from nuronet2.base import get_regulariser
from nuronet2.backend import N
import numpy


        
if __name__ == "__main__":
    model = nuro.NeuralNetwork()
    model.add(TemporalDense(8, input_shape=(None, 1)))
    
    data = numpy.random.randn(32, 64, 1)
    y = model.predict(data)