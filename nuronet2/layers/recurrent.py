# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:20:04 2017

@author: Evander
"""

import numpy
from nuronet2.backend import N
from nuronet2.base import Layer, InputDetail, get_weightfactory, get_regulariser
from nuronet2.activations import get_activation


def time_dist_dense(x, w, b, is_training, dropout=None,
                    input_dim=None, output_dim=None, 
                    timesteps=None):
    """
    Apply 'W.y + b' for every temporal slice y of x
    
    Inputs
    ------
        @param x : tensor holding time series data
        @param w : weight matrix
        @param b : bias vector
        @param is_training: is the caller in training phase or not
        @param dropout : applies dropout to the operation
        @param input_dim: (optional) dimensionality of the input
        @param output_dim: (optional) dimensionality of the output
        @param timesteps: (optional) number of timesteps
        
    Returns
    -------
        Output tensor
    """
    if(not input_dim):
        input_dim = N.shape(x)[2]
    if(not timesteps):
        timesteps = N.shape(x)[1]
    if(not output_dim):
        output_dim = N.shape(w)[1]
    
    if(dropout is not None and 0. < dropout < 1.):
        ones = N.ones_like(N.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = N.dropout(ones, dropout)
        expanded_dropout_matrix = N.repeat(dropout_matrix, timesteps)
        if(is_training):
            x = x * expanded_dropout_matrix
            
    #collpse time dimension and batch dimension together
    x = N.reshape(x, (-1, input_dim))
    x = N.dot(x, w)
    x += b
    
    #reshape to 3D
    if N.backend() == 'tensorflow':
        x = N.reshape(x, N.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = N.reshape(x, (-1, timesteps, output_dim))
    return x
    
    
class Recurrent(Layer):
    """
    Base class for recurrent layers
    """
    def __init__(self, return_sequences=True, go_backwards=False,
                 unroll=False, input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.unroll = unroll
        self.input_dim = input_dim
        self.input_length = input_length
        if(self.input_dim):
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        Layer.__init__(self, **kwargs)
        
    def get_output_shape(self, input_shape):
        if(self.return_sequences):
            return (input_shape[0], input_shape[1], self.n)
    
    def make_initial_states(self, x):
        # build an all zero tensor of shape (samples, output_dim)
        initial_state = N.zeros_like(x)  #(samples, timesteps, input_dim)
        initial_state = N.sum(initial_state, axis=(1, 2)) #(samples, )
        initial_state = N.expand_dim(initial_state) #(samples, 1)
        initial_state = N.tile(initial_state, [1, self.n]) #(samples, output_dim)
        return [initial_state for _ in range(len(self.states))]
        
    def prop_up(self, x):
        input_shape = N.int_shape(x)
        if(self.unroll and input_shape[1] is None):
            raise ValueError('Cannot unroll an RNN if the time dimension'
                            'isn\'t specified.')
        initial_states = self.make_initial_states(x)
        constants = self.get_constants(x)
        
        last_output, outputs, states = N.recurrence(self._step,
                                                    x,
                                                    initial_states,
                                                    go_backwards=self.go_backwards,
                                                    constants=constants,
                                                    unroll=self.unroll,
                                                    input_length=input_shape[1])
        if(self.return_sequences):
            return outputs
        else:
            return last_output
        
    def get_constants(self, x):
        raise NotImplementedError()
        
    def build(self, input_shape):
        raise NotImplementedError()
        
    def get_cost(self):
        raise NotImplementedError()
        
    def _step(self, x, states):
        raise NotImplementedError()
        
        

class RNNLayer(Recurrent):
    """
    Implements a simple RNN
    
    Inputs
    ------
        @param n: The number of hidden units
        @param weight_factory: how to initialise input -> hidden weights
        @param h_factory: how to initialise hidden -> hidden weights
        @param activation: nonlinear activation function of the hidden units
        @param w_regulariser: weight regulariser for input->hidden weights
        @param h_regulariser: weight regulariser for hidden->hidden weights
    """
    def __init__(self, n, weight_factory='xavier_uniform',
                 h_factory='orthogonal', activation='tanh',
                 w_regulariser=None, h_regulariser=None, b_regulariser=None,
                 dropout_w=0., dropout_h=0., **kwargs):
        self.n = n
        self.w_factory = get_weightfactory(weight_factory)
        self.h_factory = get_weightfactory(h_factory)
        self.activation = get_activation(activation)
        self.w_regulariser = get_regulariser(w_regulariser)
        self.b_regulariser = get_regulariser(b_regulariser)
        self.h_regulariser = get_regulariser(h_regulariser)
        self.dropout_w = dropout_w
        self.dropout_h = dropout_h
        
        Recurrent.__init__(self, **kwargs)
        
    def build(self, input_shape):
        self.input_details = [InputDetail(shape=input_shape)]
        self.input_dim = input_shape[2]
        self.states = [None]
        
        self.W = self.w_factory(shape=(self.input_dim, self.n))
        self.b = N.zeros(shape=(self.n,))
        self.H = self.h_factory(shape=(self.n, self.n))
        self.trainable_weights = [self.W, self.b, self.H]
        
        self.built = True
        
    def get_cost(self):
        w_cost =  self.w_regulariser(self.W) if self.w_regulariser else N.cast(0.)
        b_cost = self.b_regulariser(self.b) if self.b_regulariser else N.cast(0.)
        h_cost =  self.h_regulariser(self.H) if self.h_regulariser else N.cast(0.)
        return w_cost + b_cost + h_cost

    def _step(self, x, states):
        prev_output = states[0]
        B_H = states[1]
        B_W = states[2]
        h = N.dot(x * B_W, self.W) + self.b
        output = self.activation(h + N.dot(prev_output * B_H, self.H))
        return output, [output]
        
    def get_constants(self, x):
        constants = []
        if(0 < self.dropout_h < 1):
            ones = N.ones_like(N.reshape(x[:,0, 0], (-1, 1)))
            ones = N.tile(ones, (1, self.n))
            if(self.is_training):
                B_H = N.dropout(ones, self.dropout_h)
            else:
                B_H = ones
            constants.append(B_H)
        else:
            constants.append(numpy.asarray(1., dtype=N.floatx))
        
        if(0 < self.dropout_w < 1):
            input_shape = N.int_shape(x)
            input_dim = input_shape[-1]
            ones = N.ones_like(N.reshape(x[:, 0, 0], (-1, 1)))
            ones = N.tile(ones, (1, int(input_dim)))
            if(self.is_training):
                B_W = N.dropout(ones, self.dropout_w)
            else:
                B_W = ones
            constants.append(B_W)
        else:
            constants.append(numpy.asarray(1., dtype=N.floatx))
        return constants
        
        

if __name__ == "__main__":
    r = Recurrent(input_dim=3)
    