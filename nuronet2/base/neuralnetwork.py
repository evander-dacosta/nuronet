
from nuronet2.base import MLModel, NetworkModel, MLConnection, get_source_inputs
from layer import Layer, InputLayer, Input

class NeuralNetwork(MLModel):
    def __init__(self, layers=None, name=None, **kwargs):
        self.layers = []
        self.model = None #internal model instance
        self.inputs = []
        self.outputs = []
        self._trainable = True #required?
        
        MLModel.__init__(self, **kwargs)
        
        self._trainable_weights = None
        self._non_trainable_weights = None
        
        
        #self.add(InputLayer(input_shape))
        if(layers):
            for layer in layers:
                self.add(layer)
                
    @property
    def trainable_weights(self):
        weights = []
        for layer in self.layers:
            if(layer.trainable):
                weights += layer.trainable_weights
        return weights
        
    @trainable_weights.setter
    def trainable_weights(self, value):
        self._trainable_weights = value
        
    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.layers:
            if(not layer.trainable):
                weights += layer.trainable_weights + layer.non_trainable_weights
        return weights

    @non_trainable_weights.setter
    def non_trainable_weights(self, value):
        self._non_trainable_weights = value
                
    def add(self, layer):
        """
        Adds a layer to the neural network's layer stack
        
        Inputs
        ------
            @param layer : A layer instance
        """
        if(not isinstance(layer, Layer)):
            raise TypeError("The added layer must be an instance "
                            "of class Layer. Found {}".format(type(layer)))
        if(not self.outputs): 
            if(len(layer.inbound_connections) == 0):
                #create an input layer
                if(not hasattr(layer, 'input_shape') or layer.input_shape is None):
                    raise ValueError("The first layer in a NeuralNetwork "
                                     "model must have an 'input_shape'")
                input_shape = layer.input_shape
                self.add(InputLayer(input_shape=input_shape))
                self.add(layer)
                return
            
            if(len(layer.inbound_connections) != 1):
                raise ValueError("The layer added to NeuralNetwork model "
                                "must not be connected elsewhere."
                                "Receiver layer {}".format(layer.name) + \
                                " which has " + \
                                str(len(layer.inbound_connections)) +\
                                " inbound connections")
            if(len(layer.inbound_connections[0].output_tensors) != 1):
                raise ValueError("The layer added to NeuralNetwork "
                                "must have a  single output tensor."
                                " Use a different API for multi-output layers")
            
            self.outputs = [layer.inbound_connections[0].output_tensors[0]]
            self.inputs = get_source_inputs(self.outputs[0])
            
            MLConnection(outbound_model=self,
                        inbound_models=[],
                        connection_indices=[],
                        tensor_indices=[],
                        input_tensors=self.inputs,
                        output_tensors=self.outputs,
                        input_shapes=[x._nuro_shape for x in self.inputs],
                        output_shapes=[self.outputs[0]._nuro_shape])
        else:
            output_tensor = layer(self.outputs[0])
            if(isinstance(output_tensor, list)):
                raise ValueError("The layer added to NeuralNetwork "
                                "must have a  single output tensor."
                                " Use a different API for multi-output layers")
            self.outputs = [output_tensor]
            self.inbound_connections[0].output_tensors = self.outputs
            self.inbound_connections[0].outputs_shapes = [self.outputs[0]._nuro_shape]
            
        self.layers.append(layer)
        self.is_built = False
        
    
    def pop(self):
        """
        Pops the last layer in the network
        """
        if(not self.layers):
            raise TypeError("There are no layers to be popped")
        self.layers.pop()
        if(not self.layers):
            self.outputs = []
            self.inbound_connections = []
            self.outbound_connections = []
        else:
            self.layers[-1].outbound_connections = []
            self.outputs = [self.layers[-1].output]
            #update self.inbound_connections
            self.inbound_connections[0].output_tensors = self.outputs
            self.inbound_connections[0].output_shapes = [self.outputs[0]._nuro_shape]
        self.is_built = False
        
    def get_layer(self, index=None):
        """
        Returns a layer by index. Indices are bottom-up
        
        Inputs
        ------
            @param name (optional) : Name of the layer required
            @param index (optional) : Index of the layer required
        
        Returns
        -------
            Layer instance
        """
        if(not self.is_built):
            self.build()
        return self.model.get_layer(index)
        
    ##To be implemented
                                       
    def build(self, input_shape=None):
        if(not self.inputs or not self.outputs):
            raise TypeError("NeuralNetwork could not be built. Add "
                            "some layers first")
        self.model = NetworkModel(self.inputs, self.outputs[0],
                                  name=self.name + '_model')
        self.model.trainable = self.trainable
        
        #mirror the model's attributes
        self.input_layers = self.model.input_layers
        self.input_layers_connection_indices = self.model.input_layers_connection_indices
        self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
        self.output_layers = self.model.output_layers
        self.output_layers_connection_indices = self.model.output_layers_connection_indices
        self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
        self.connections_by_depth = self.model.connections_by_depth
        self.container_connections = self.model.container_connections
        
        self.is_built = True
        
    def __call__(self, x=None):
        if(x is None):
            x = self.layers[0].get_input_tensors()
        if(not self.is_built):
            self.build()
        return MLModel.__call__(self, x)
        
    def prop_up(self, x=None):
        if(x is None):
            raise Exception("No argument passed to NeuralNetwork.prop_up(x). "
                            "Call NeuralNetwork() instead.")
        return self.model.prop_up(x)
        
    def get_cost(self):
        if(not self.is_built):
            self.build()
        return self.model.get_cost()
        
    def get_output_shape(self, input_shape=None):
        if(not self.is_built):
            self.build()
        return self.model.get_output_shape(input_shape)
        
    def get_updates(self):
        if(not self.is_built):
            self.build()
        return self.model.get_updates()
        
    def set_training(self, value):
        if(not self.is_built):
            self.build()
        return self.model.set_training(value)
        
    def get_weights(self):
        if(not self.is_built):
            self.build()
        return self.model.get_weights()
        
    def set_weights(self, weights):
        if(not self.is_built):
            self.build()
        return self.model.set_weights(weights)
        
    def save_weights(self, filepath, overwrite=True):
        if(not self.is_built):
            self.build()
        return self.model.save_weights(filepath, overwrite)
    
    def load_weights(self, filepath, by_name=False):
        if(not self.is_built):
            self.build()
        return self.model.load_weights(filepath, by_name)
        
    def save_weights_to_hdf5_group(self, f):
        if(not self.is_built):
            self.build()
        return self.model.save_weights_to_hdf5_group(f)
        
    def load_weights_from_hdf5_group(self, f):
        if(not self.is_built):
            self.build()
        return self.model.load_weights_from_hdf5_group(f)
        
    def load_weights_from_hdf5_group_by_name(self, f):
        if(not self.is_built):
            self.build()
        return self.model.load_weights_from_hdf5_group_by_name(f)
    
    def predict(self, input_values):
        if(not self.is_built):
            self.build()
        return self.model.predict(input_values)
        
    def get_predictor(self):
        if(not self.is_built):
            self.build()
        return self.model.get_predictor()
        

        
    
        
        
