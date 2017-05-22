import numpy
from nuronet2.base import InputDetail
from nuronet2.activations import get_activation
from nuronet2.backend import N
from nuronet2.base import get_weightfactory, get_regulariser, Layer



        
if __name__ == "__main__":
    import nuronet2 as nuro
    
    model = nuro.NeuralNetwork()
    model.add(ConvDenseLayer(64, input_shape=(1, None)))
    
    data = numpy.random.randn(32, 1, 100)
    x = model.predict(data)