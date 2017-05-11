#
from utils import *

"""
TODO: Find a way to import the default backend N from config files or
      some crazy initial setup
"""
flags = ['use_theano', 'use_tensorflow']
flag = 'use_theano'
N = None


if(flag == 'use_theano'):
    from theanobackend import TheanoBackend
    N = TheanoBackend(default_dtype='float32')
    
elif(flag == 'use_tensorflow'):
    from tensorflowbackend import TensorflowBackend
    N = TensorflowBackend()