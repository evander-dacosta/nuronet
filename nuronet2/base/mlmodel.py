# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:54:21 2016

@author: evander
"""

from nuronet2.backend import N
from collections import OrderedDict

def make_list(x):
    """Normalises a list/tensor into a list
    """
    if(type(x) is list):
        return x
    return [x]


class InputDetail(object):
    """Specifies the ndim, dtype, and shape of every input to an MLModel.
    Every MLModel should expose an input_details attribute: A list of 
    InputDetail objects, one for every input
    """
    
    def __init__(self, ndim=None, dtype=None, shape=None):
        if(ndim is not None):
            if(not isinstance(ndim, int)):
                raise Exception("ndim has to be int. Given {}".format(type(ndim)))
            self.ndim = ndim
            
        if(shape is not None):
            self.ndim = len(shape)
        self.dtype = dtype
        self.shape = shape
        

class MLConnection(object):
    """Describes connectivity between MLModel instances
    
    Everytime an MLModel is connected to some new input, an MLConnection
    is added to MLModel.inbound_connections
    Everytime the output of an MLModel is used by another MLModel,
    a node is added to MLModel.outbound_connections
    
    outbound_model: The model that takes 'input_tensors' and turns them into
                    'output_tensors'
    inbound_models: A list of models, with the same length as 'input_tensors',
                    the models from which these tensors originate
    connection_indices: A list of integers with the same length as 'inbound_models,
                        connection_indices[i] is the origin connection of 
                        'input_tensors[i]' (since each inbound_model might have
                        multiple connections)
    tensor_indices: A list of integers, with the same length as 'inbound_models',
                    'tensor_indices[i]' is the index of 'input_tensors[i]',
                    within the output of the inbound model (since inbound
                    models might have multiple outputs)
    input_tensors: A list of tensors
    output_tensors: A list of output tensors
    input_shapes: A list of input shape tuples
    output_shapes: A list of output shape tuples
    
    A connection from A to B is added to:
    A.outbound_connections
    B.inbound_connections
    """
    
    def __init__(self, outbound_model, inbound_models, connection_indices,
                 tensor_indices, input_tensors, output_tensors, input_shapes,
                 output_shapes):
        self.outbound_model = outbound_model
        
        #The following describe where the input tensors
        #came from: which models, and for each model, 
        #which connection, and which tensor output of each
        #connection
        
        self.inbound_models = inbound_models
        self.connection_indices = connection_indices 
        self.tensor_indices = tensor_indices
        
        #tensor inputs and outputs of outbound_model
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        
        #add connections to all models involved
        for model in self.inbound_models:
            if(model is not None):
                model.outbound_connections.append(self)
        outbound_model.inbound_connections.append(self)
        
        
    @classmethod
    def create_connection(cls, outbound_model, inbound_models, 
                          connection_indices=None, tensor_indices=None):
        if(not connection_indices):
            connection_indices = [0 for _ in range(len(inbound_models))]
        else:
            assert len(connection_indices) == len(inbound_models)
            
        if(not tensor_indices):
            tensor_indices = [0 for _ in range(len(inbound_models))]
        
        input_tensors = []
        input_shapes = []
        
        for inbound_model, connection_index, tensor_index in zip(inbound_models, connection_indices, tensor_indices):
            inbound_connection = inbound_model.inbound_connections[connection_index]
            input_tensors.append(inbound_connection.output_tensors[tensor_index])
            input_shapes.append(inbound_connection.output_shapes[tensor_index])
            
        assert len(input_shapes) == len(input_tensors)
        
        if(len(input_tensors) == 1):
            output_tensors = make_list(outbound_model.prop_up(input_tensors[0]))
            output_shapes = make_list(outbound_model.get_output_shape(input_shapes[0]))
        else:
            output_tensors = make_list(outbound_model.prop_up(input_tensors))
            output_shapes = make_list(outbound_model.get_output_shape(input_shapes))
            
        if(not output_tensors or output_tensors[0] is None):
            raise Exception("The 'call' method of model " + outbound_model.name +
                            " should return a tensor. Found: " + str(output_tensors[0]))
        if(len(output_tensors) != len(output_shapes)):
            raise Exception("The get_output_shape_for method should return one shape tuple"+\
                            " per output tensor of the layer.")
                            
        for i in range(len(output_tensors)):
            output_tensors[i]._nuro_shape = output_shapes[i]
            output_tensors[i]._nuro_history = (outbound_model, len(outbound_model.inbound_connections), i)
            
        return cls(outbound_model, inbound_models, connection_indices,
                   tensor_indices, input_tensors, output_tensors, input_shapes,
                   output_shapes)
                   
                                      
class MLModel(object):
    def __init__(self, **kwargs):
        self.input_details = None
        self.inbound_connections = []
        self.outbound_connections = []
        self.regularisers = []
        self.updates = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.is_built = False
        self.train_phase = True
        
        valid_kwargs = {'input_shape', 'input_dtype', 'name'}
        for kwarg in kwargs.keys():
            assert kwarg in valid_kwargs, 'Keyword arg not recognised: ' + kwarg
        
        name = kwargs.get('name')
        if(not name):
            name = '_' + self.__class__.__name__.lower() + '_'
            name = name + str(N.get_uid(name))
        self.name = name
        
        if('input_shape' in kwargs):
            self.input_shape = (None, ) + tuple(kwargs['input_shape'])
        else:
            self.input_shape = None
        self.input_dtype = kwargs.get('input_dtype')
        
    @property
    def weights(self):
        return self.trainable_weights + self.non_trainable_weights
        
    def __call__(self, x):
        if(not self.is_built):
            input_shapes = []
            for elem in make_list(x):
                if(hasattr(elem, '_nuro_shape')):
                    input_shapes.append(elem._nuro_shape)
                elif(hasattr(N, 'int_shape')):
                    input_shapes.append(N.int_shape(elem))
                else:
                    raise ValueError("Model " + self.name + " has no information " + \
                    "about its expected input's shape and cannot be built or called")
            if(len(input_shapes) == 1):
                self.build(input_shapes[0])
            else:
                self.build(input_shapes)
            self.is_built = True

        input_tensors = make_list(x)
        inbound_models = []
        connection_indices = []
        tensor_indices = []
        for input_tensor in input_tensors:
            if(hasattr(input_tensor, '_nuro_history') and input_tensor._nuro_history):
                previous_model, connection_index, tensor_index = input_tensor._nuro_history
                inbound_models.append(previous_model)
                connection_indices.append(connection_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_models = None
                break
            
        
        if(inbound_models):
            self.add_inbound_connection(inbound_models, connection_indices, tensor_indices)
            #outputs were already computed when calling add_inbound_connection
            outputs = self.inbound_connections[-1].output_tensors
            #If single output tesnor: return that tensor
            #else return a list
            if(len(outputs) == 1):
                return outputs[0]
            else:
                return outputs
                
        else:
            return self.prop_up(x)
            
    def add_inbound_connection(self, inbound_models, connection_indices=None, 
                               tensor_indices=None):
        inbound_models = make_list(inbound_models)
        if(not connection_indices):
            connection_indices = [0 for _ in range(len(inbound_models))]
        else:
            connection_indices = make_list(connection_indices)
            assert len(connection_indices) == len(inbound_models)
        
        if(not tensor_indices):
            tensor_indices = [0 for _ in range(len(inbound_models))]
        else:
            tensor_indices =  make_list(tensor_indices)
            
        if(not self.is_built):
            input_shapes = []
            for model, connection_index, tensor_index in zip(inbound_models, connection_indices, tensor_indices):
                input_shapes.append(model.inbound_connections[connection_index].output_shapes[tensor_index])
                
            if(len(input_shapes) == 1):
                self.build(input_shape=input_shapes[0])
            else:
                self.build(input_shape=input_shapes)
            self.is_built = True
        MLConnection.create_connection(self, inbound_models, connection_indices,
                                       tensor_indices)
                                       
                                       
    def get_connection_attribute_at(self, index, attr):
        if(not self.inbound_connections):
            raise Exception("This model has never been called and therefore has no defined {}.".format(attr))
        if(not len(self.inbound_connections) > index):
            raise Exception("index greater than number of connections")
        values = getattr(self.inbound_connections[index], attr)
        if(len(values) == 1):
            return values[0]
        return values
                                       
                                       
    def get_input_shape_at(self, connection_index):
        return self.get_connection_attribute_at(connection_index, 'input_shapes')
        
    def get_output_shape_at(self, connection_index):
        return self.get_connection_attribute_at(connection_index, 'output_shapes')
        
    def get_input_at(self, index):
        return self.get_connection_attribute_at(index, 'input_tensors')
        
    def get_output_at(self, index):
        return self.get_connection_attribute_at(index, 'output_tensors')
        
    def set_weights(self, weights):
        params = self.weights
        if(len(params) != len(weights)):
            raise ValueError("Tried to set {} weights but the model was expecting".format(len(weights)) +\
                            " {} weights".format(len(params)))
        if(not params):
            return
        
        param_values = [N.get_value(param) for param in params]
        for pv, p, w in zip(param_values, params, weights):
            if(pv.shape != w.shape):
                raise ValueError("Model weight shape not compatible with given weight shape")
            N.set_value(p, w)
            
    def get_weights(self):
        params = self.weights
        return [N.get_value(param) for param in params]


                                       
                                       
    ##To be implemented
                                       
    def build(self, input_shape):
        raise NotImplementedError()
        
    def prop_up(self, x):
        return x
        
    def get_cost(self):
        raise NotImplementedError()
        
    def get_output_shape(self, input_shape):
        """Computes the output shape of the layer
        given its input shape
        """
        return input_shape
        
    def get_updates(self):
        return OrderedDict()
    