#
from utils import *

"""
TODO: Find a way to import the default backend N from config files or
      some crazy initial setup
"""
flags = ['use_theano', 'use_tensorflow']
flag = 'use_theano'
N = None

def use_theano():
    global N
    from theanobackend import TheanoBackend
    N = TheanoBackend(default_dtype='float32')
    
def use_tensorflow():
    global N
    from tensorflowbackend import TensorflowBackend
    N = TensorflowBackend()
    

if(flag == 'use_theano'):
    print "Using theano"
    use_theano()
    
elif(flag == 'use_tensorflow'):
    print "Using tensorflow"
    use_tensorflow()

    
    
