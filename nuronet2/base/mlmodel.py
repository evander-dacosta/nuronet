# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:54:21 2016

@author: evander
"""
import numpy
import time
import threading
import multiprocessing

from nuronet2.backend import N
from collections import OrderedDict
from nuronet2.dataset import DenseDataset
from nuronet2.objectives import get_objective

from nuronet2.optimisers import get_optimiser
import nuronet2.callbacks as cbks

try:
    import queue
except ImportError:
    import Queue as queue

def make_list(x):
    """Normalises a list/tensor into a list
    """
    if(type(x) is list):
        return x
    return [x]
    
def get_source_inputs(tensor, layer=None, connection_index=None):
    """
    Returns a list of input sources necessary to compute 
    'tensor'
    
    Inputs
    ------
        @param tensor : The tensor to compute
        @param layer : Layer instance from which the tensor originates
        @param connection_index: Origin MLConnection index of the tensor
    """
    if(not hasattr(tensor, '_nuro_history')):
        return tensor
    
    if(layer is None or connection_index):
        layer, connection_index, _ = tensor._nuro_history
    if(not layer.inbound_connections):
        return [tensor]
    else:
        connection = layer.inbound_connections[connection_index]
        if(not connection.inbound_models):
            #reached an input layer, stop.
            return connection.input_tensors
        else:
            source_tensors = []
            for i in range(len(connection.inbound_models)):
                x = connection.input_tensors[i]
                layer = connection.inbound_models[i]
                connection_index = connection.connection_indices[i]
                previous_sources = get_source_inputs(x, layer, 
                                                     connection_index)
                for x in previous_sources:
                    if(x not in source_tensors):
                        source_tensors.append(x)
            return source_tensors


class InputDetail(object):
    """Specifies the ndim, dtype, and shape of every input to an MLModel.
    Every MLModel should expose an input_details attribute: A list of 
    InputDetail objects, one for every input
    
    Inputs
    ------
        @param ndim (optional) : The number of dimensions in the input tensor
        @param dtype : The input's data type
        @param shape (optional):
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
    
    Inputs
    ------
    
        @param outbound_model: The model that takes 'input_tensors' and turns them into
                        'output_tensors'
        @param inbound_models: A list of models, with the same length as 'input_tensors',
                        the models from which these tensors originate
        @param connection_indices: A list of integers with the same length as 'inbound_models,
                            connection_indices[i] is the origin connection of 
                            'input_tensors[i]' (since each inbound_model might have
                            multiple connections)
        @param tensor_indices: A list of integers, with the same length as 'inbound_models',
                        'tensor_indices[i]' is the index of 'input_tensors[i]',
                        within the output of the inbound model (since inbound
                        models might have multiple outputs)
        @param input_tensors: A list of tensors
        @param output_tensors: A list of output tensors
        @param input_shapes: A list of input shape tuples
        @param output_shapes: A list of output shape tuples
    
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
        
        for inbound_model, connection_index, tensor_index in \
        zip(inbound_models, connection_indices, tensor_indices):
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
        self.is_training = False
        self.trainable = True
        
        valid_kwargs = {'input_shape', 'input_dtype', 'name'}
        for kwarg in kwargs.keys():
            assert kwarg in valid_kwargs, 'Keyword arg not recognised: ' + kwarg
        
        name = kwargs.get('name')
        if(not name):
            name = '_' + self.__class__.__name__.lower() + '_'
            name = name + str(N.get_uid(name))
        self.name = name
        
        if('input_shape' in kwargs or 'batch_input_shape' in kwargs):
            if('batch_input_shape' in kwargs):
                batch_input_shape = tuple(kwargs['batch_input_shape'])
                input_shape = batch_input_shape[1:]
            elif('input_shape' in kwargs):
                if('batch_size' in kwargs):
                    batch_size = kwargs['batch_size']
                else:
                    batch_size = None
                input_shape = tuple(kwargs['input_shape'])
                batch_input_shape = (batch_size, ) + input_shape
            self.batch_input_shape = batch_input_shape
            self.input_shape = input_shape

        self.input_dtype = kwargs.get('input_dtype')
        
    def set_training(self, value):
        assert isinstance(value, bool), "set_training requires True/False arg"
        self.is_training = value
        
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
        
    def compile(self, optimiser, loss, **kwargs):
        """
        Configures the model for training.
        
        Inputs
        ------
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
            loss: str (name of objective function) or objective function.
                See [objectives](/objectives).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of objectives.
            kwargs: when using the Theano backend, these arguments
                are passed into N.function. Ignored for Tensorflow backend.
        """
        self.build()
        if(not isinstance(loss, list)):
            loss = [loss]
        self.optimiser = get_optimiser(optimiser)
        self.loss = loss
        
        if(len(loss) != len(self.outputs)):
            raise ValueError("loss should have one entry per output "
                            "currently has {} entries".format(len(loss)) +
                            "for {} outputs".format(len(self.outputs)))
        self.loss_functions = [get_objective(objective) for objective in loss]
        
        #prepare targets of model
        self.targets = []
        for i in range(len(self.outputs)):
            shape = self.outputs[i]._nuro_shape
            name = self.output_layers[i].name
            self.targets.append(N.variable(ndim=len(shape),
                                           dtype=N.dtype(self.outputs[i]),
                                            name='h'+ name + '_target'))
        #compute total loss
        total_loss = None
        for i in range(len(self.outputs)):
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            loss_fn = self.loss_functions[i]
            if(total_loss is None):
                total_loss = loss_fn(y_true, y_pred)
            else:
                total_loss += loss_fn(y_true, y_pred)
        
        #add regularisation penalties
        total_loss += self.get_cost()
        
        #prepare gradient updates and state updates
        self.total_loss = total_loss
        
        # functions for train and test will
        # be created when required
        self._function_kwargs = kwargs
        
        self.train_function = None
        self.test_function = None
        
        
        
    def make_train_function(self):
        if(not(hasattr(self, 'train_function'))):
            raise RuntimeError('Model must be compiled before a'
                                'train function is made')
        if(self.train_function is None):
            inputs = self.inputs + self.targets
            
            training_updates = self.optimiser.get_updates(self.trainable_weights,
                                             self.total_loss)
            updates = self.get_updates().items() + training_updates
        
            self.train_function = N.function(inputs, 
                                         [self.total_loss],
                                         updates=updates,
                                         **self._function_kwargs)
    def make_test_function(self):
        if(not(hasattr(self, 'test_function'))):
            raise RuntimeError('Model must be compiled before a'
                                'test function is made')
        if(self.test_function is None):
            inputs = self.inputs + self.targets
            self.test_function = N.function(inputs,
                                            [self.total_loss],
                                            **self._function_kwargs)
    
    def fit(self, x, y, batch_size=32, n_epochs=10, verbose=True,
            callbacks=None, validation_split=0.1, test_data=None,
            shuffle=True, initial_epoch=0):
        """
        Trains the model for a fixed number of epochs
        
        Inputs
        ------
            @param x: input training data
            @param y: output training data (list of outputs if the model
                      has multiple outputs)
            @param batch_size: batch_size
            @param n_epochs: Total number of epochs
            @param callbacks: List of callbacks
            @param validation_split: float between 0 and 1 representing the 
                                    proportion of dataset to set aside as
                                    validation
            @param validation_data: data on which to evaluate loss and model
                                    metrixs
            @param initial_epoch: epoch at which to start training
                                    (useful for resuming from previous runs)
        
        Returns
        -------
            History instance. This contains all information 
            collected during training
        """
        assert(self.is_built)
        #prepare test data
        if(test_data is not None):
            if(len(test_data) == 2):
                x_test, y_test = test_data
            else:
                raise ValueError("test_data supplied to model.fit() must "
                                "be a tuple/list of size 2 e.g. [x_test, y_test]")
        else:
            x_test = None
            y_test = None
            
        #create dataset
        dataset = DenseDataset(x=x, y=y, x_test=x_test,
                               y_test=y_test, batch_size=batch_size,
                               validation=validation_split,
                               shuffle=shuffle)
        
        
        return self.fit_dataset(dataset=dataset, n_epochs=n_epochs, 
                                callbacks=callbacks,
                                verbose=verbose, initial_epoch=initial_epoch)
                                
    def fit_generator(self, generator, n_epochs,
                      verbose=True, callbacks=None, max_q_size=10,
                      n_workers=1, pickle_safe=False, initial_epoch=0):
        """
        Fits the model on data generated batch-by-batch by a python
        generator. The generator runs in parallel ith the model.
        
        For example, the generator allows us to do data augmentation on the 
        CPU while the model trains on the GPU
        
        Inputs
        ------
            @param generator: A generator dataset type
            @param n_epochs: Total number of epochs
            @param verbose : Print out per-epoch metrics?
            @param callbacks: List of callbacks
            @param max_q_size: Maximum size for the generator queue
            @param n_workers: Number of processes to spin up when using 
                             process-based threading
            @param pickle_safe: If True, use process-based threading
            @param initial_epoch: Epoch at which to start training
        Returns
        -------
            History instance. This contains all information 
            collected during training 
        """
        wait_time = 0.01 #seconds
        epoch = initial_epoch
        self.make_train_function()
        self.make_test_function()
        
        self.history = cbks.History()
        callbacks = (callbacks or []) + [self.history]
        if(verbose):
            callbacks += [cbks.TrainLogger()]
        callbacks = cbks.CallbackList(callbacks)
        
        #make it possible to call callbacks from a different model
        if(hasattr(self, 'callback_model') and self.callback_model):
            callback_model = self.callback_model
        else:
            callback_model = self
            
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'batch_size': generator.batch_size,
            'n_epochs': n_epochs,
            'verbose': verbose
        })
        callbacks.train_start()
        
        enqueuer = None
        
        try:
            enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
            enqueuer.start(max_q_size=max_q_size, n_workers=n_workers)
            
            while(epoch < n_epochs):
                callbacks.epoch_start(epoch)
                epoch_logs = {}
                epoch_start_time = time.time()
                
                samples_seen = 0
                batch_index = 0
                epoch_loss = []
                
                #Make the generator generate indices for training and validation
                generator.make_validation_splits()
    
                while(samples_seen < generator.n):
                    generator_output = None
                    while enqueuer.is_running():
                        if not enqueuer.queue.empty():
                            generator_output = enqueuer.queue.get()
                            break
                        else:
                            time.sleep(wait_time)
                            
                    
                    x, y = generator_output
                    
                    #build batch logs
                    batch_logs = {}
                    batch_size = x.shape[0]
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = batch_size
                    callbacks.batch_start(batch_index, batch_logs)
                    loss = self.train_function(x, y)
                    epoch_loss += loss
                    batch_logs['loss'] = loss[0]
                    callbacks.batch_end(batch_index, batch_logs)
                    
                    
                    batch_index += 1
                    samples_seen += batch_size
                #do validation here
                if(len(generator.valid_indices) > 0):
                    valid_loss = self.test_function(generator.x_valid,
                                                    generator.y_valid)
                    epoch_logs['valid_loss'] = valid_loss[0]
                epoch_logs['epoch'] = epoch
                epoch_logs['train_loss'] = numpy.mean(epoch_loss)
                epoch_logs['epoch_time'] = time.time() - epoch_start_time
                callbacks.epoch_end(epoch, epoch_logs)
                epoch += 1
        finally:
            if(enqueuer is not None):
                enqueuer.stop()
        callbacks.train_end()
        return self.history
        
                                
    def fit_dataset(self, dataset, n_epochs, callbacks=None, verbose=True, 
                     initial_epoch=0):
        
        self.make_train_function()
        self.make_test_function()

        #initialise callbacks
        self.history = cbks.History()
        callbacks = (callbacks or []) + [self.history]
        if(verbose):
            callbacks += [cbks.TrainLogger()]
        callbacks = cbks.CallbackList(callbacks)
        
        #make it possible to call callbacks from a different model
        if(hasattr(self, 'callback_model') and self.callback_model):
            callback_model = self.callback_model
        else:
            callback_model = self
        
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'batch_size': dataset.batch_size,
            'n_epochs': n_epochs,
            'verbose': verbose
        })
        callbacks.train_start()
        
        for epoch in range(initial_epoch, n_epochs):
            callbacks.epoch_start(epoch)
            epoch_start_time = time.time()
            epoch_logs = {}
            
            #Make the generator generate indices for training and validation
            dataset.make_validation_splits()
            
            
            #iterate over batches
            batch_index = 0
            epoch_loss = []
            valid_loss = 0
            samples_seen = 0
            while(samples_seen < dataset.n):
                batch_index += 1
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = dataset.batch_size
                
                callbacks.batch_start(batch_index, batch_logs)
                x_b, y_b = dataset.next()
                loss = self.train_function(x_b, y_b)
                epoch_loss += loss
                batch_logs['loss'] = loss[0]
                epoch_loss += loss
                callbacks.batch_end(batch_index, batch_logs)
                
                #increment counters
                batch_index += 1
                samples_seen += x_b.shape[0]
                
            valid_loss = self.test_function(dataset.x_valid, 
                                            dataset.y_valid)
            epoch_logs['epoch'] = epoch
            epoch_logs['train_loss'] = numpy.mean(epoch_loss)
            epoch_logs['valid_loss'] = valid_loss[0]
            epoch_logs['epoch_time'] = time.time() - epoch_start_time
            callbacks.epoch_end(epoch, epoch_logs)
        
        callbacks.train_end()
        return self.history
        
        
        
        
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
        
        
        
        


class GeneratorEnqueuer(object):
    """
    Builds a queue out of a data generator
    
    Inputs
    ------
        @param generator: A generator dataset instance
        @param pickle_safe: use multiprocessing if True, else use threading
    """
    def __init__(self, generator, pickle_safe=False):
        self._generator = generator
        self._pickle_safe = pickle_safe
        self._threads = []
        self._stop_event = None
        
        self.queue = None
        
    def start(self, n_workers=1, max_q_size=10, wait_time=0.05):
        """
            Kick off threads which add data from the generator into the queue.
            Inputs
            ------
                @param n_workers: number of worker threads
                @param max_q_size: queue size (when full, threads could block on put())
                @param wait_time: time to sleep in-between calls to put()
        """
        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if(self._pickle_safe or self.queue.qsize() < max_q_size):
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    self._stop_event.set()
                    raise
        try:
            if(self._pickle_safe):
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()
            for i in range(n_workers):
                if(self._pickle_safe):
                    #reset seed so children don't share same seed
                    numpy.random.seed()
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise
            
    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()
        
    def stop(self, timeout=None):
        """
        Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().
        """
        if(self.is_running()):
            self._stop_event.set()
        for thread in self._threads:
            if(thread.is_alive()):
                if(self._pickle_safe):
                    thread.terminate()
                else:
                    thread.join(timeout)
                    
        if(self._pickle_safe):
            if(self.queue is not None):
                self.queue.close()
        self._threads = []
        self._stop_event = None
        self.queue = None
    