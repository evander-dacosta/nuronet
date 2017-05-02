# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:58:35 2016

@author: evander
"""

# -*- coding: utf-8 -*-

import warnings
import os
from collections import OrderedDict
from nuronet2.backend import N
from mlmodel import MLModel, MLConnection, make_list


class NetworkModel(MLModel):
    """A container for acyclically connected MLModels
    """
    def __init__(self, inputs, outputs, name=None, **kwargs):
        
        if(not name):
            name = "_" + self.__class__.__name__.lower() + "_"
            name = name + str(N.get_uid(name))
        self.name = name
        if(type(inputs) in (list, tuple)):
            self.inputs = list(inputs)
        else:
            self.inputs = [inputs]
            
        if(type(outputs) in (list, tuple)):
            self.outputs = list(outputs)
        else:
            self.outputs = [outputs]
            
        #check for input redundancy
        in_set = set(self.inputs)
        if(len(in_set) != len(self.inputs)):
            raise Exception("Redundant inputs given to model")
        
        self.input_layers = []
        self.input_layers_connection_indices = []
        self.input_layers_tensor_indices = []
        self.output_layers = []
        self.output_layers_connection_indices = []
        self.output_layers_tensor_indices = []
        
        self.layers = []
        self._output_tensor_cache = {}
        self._output_shape_cache = {}
        
        #Argument validation
        for x in self.inputs:
            if(not hasattr(x, '_nuro_history')):
                cls_name = self.__class__.__name__
                raise Exception("Input tensors to {} must be Nuronet tensors".format(cls_name) + \
                                "Found an input variable with no _nuro_history and _nuro_shape metadata.")
            layer, connection_index, tensor_index = x._nuro_history
            if(len(layer.inbound_connections) > 1 or (layer.inbound_connections and layer.inbound_connections[0].inbound_models)):
                cls_name = self.__class__.__name__
                warnings.warn(cls_name + " Inputs must come from a Nuronet Input Layer.")
                
        for x in self.outputs:
            if(not hasattr(x, '_nuro_history')):
                cls_name = self.__class__.__name__
                raise Exception("Output tensors to {} must be Nuronet tensors".format(cls_name) + \
                                "Found an input variable with no _nuro_history and _shape metadata.")
        
        #build self.output_layers
        for x in self.outputs:
            layer, connection_index, tensor_index = x._nuro_history
            self.output_layers.append(layer)
            self.output_layers_connection_indices.append(connection_index)
            self.output_layers_tensor_indices.append(tensor_index)
            
        #build self.input_layers
        for x in self.inputs:
            layer, connection_index, tensor_index = x._nuro_history
            assert connection_index == 0
            assert tensor_index == 0
            self.input_layers.append(layer)
            self.input_layers_connection_indices.append(connection_index)
            self.input_layers_tensor_indices.append(tensor_index)
            
        self.internal_input_shapes = [x._nuro_shape for x in self.inputs]
        self.internal_output_shapes = [x._nuro_shape for x in self.outputs]
        
        #set of all MLConnections in the graph
        container_connections = set()
        connection_depths = {} #{node: depth_value}
        layer_depths = {} #{layer: depth value}
        layer_indices = {} # {layer: index in traversal}
        
        def connection_marker(connection, depth):
            return str(id(connection)) + '-' + str(depth)
            
        def build_graph_map(tensor, seen=set(), depth=0,
                            layer=None, connection_index=None, tensor_index=None):
            if(not layer or connection_index is None or tensor_index is None):
                layer, connection_index, tensor_index = tensor._nuro_history
            connection = layer.inbound_connections[connection_index]
            
            seen.add(connection_marker(connection, depth))
            
            connection_key = layer.name + '_ib-'+str(connection_index)
            container_connections.add(connection_key)
            
            #update connection depths
            connection_depth = connection_depths.get(connection)
            if(connection_depth is None):
                connection_depths[connection] = depth
            else:
                connection_depths[connection] = max(depth, connection_depth)
            
            #update layer depths
            prev_seen_depth = layer_depths.get(layer)
            if(prev_seen_depth is None):
                current_depth = depth
            else:
                current_depth = max(depth, prev_seen_depth)
            layer_depths[layer] = current_depth
            if(layer not in layer_indices):
                layer_indices[layer] = len(layer_indices)
                
            #Propagate to all previous tensors connected to this connection
            for i in range(len(connection.inbound_models)):
                x = connection.input_tensors[i]
                layer = connection.inbound_models[i]
                connection_index = connection.connection_indices[i]
                tensor_index = connection.tensor_indices[i]
                next_connection = layer.inbound_connections[connection_index]
                #use connection marker to prevent cycles
                con_marker = connection_marker(next_connection, current_depth + 1)
                if(con_marker not in seen):
                    build_graph_map(x, seen, current_depth + 1, layer,
                                    connection_index, tensor_index)
                                    
        for x in self.outputs:
            seen = set()
            build_graph_map(x, seen, depth=0)
        
        connections_by_depth = {} # {depth: connections with this depth}
        for con, depth in connection_depths.items():
            if(depth not in connections_by_depth):
                connections_by_depth[depth] = []
            connections_by_depth[depth].append(con)
            
        layers_by_depth = {} #depth: layers with this depth
        for layer, depth in layer_depths.items():
            if(depth not in layers_by_depth):
                layers_by_depth[depth] = []
            layers_by_depth[depth].append(layer)
            
        #sorted list of layer depths
        depth_keys = list(layers_by_depth.keys())
        depth_keys.sort(reverse=True)

        #set self.layers and self.layers_by_depth        
        layers = []
        for depth in depth_keys:
            layers_for_depth = layers_by_depth[depth]
            layers_for_depth.sort(key=lambda x: layer_indices[x])
            for layer in layers_for_depth:
                layers.append(layer)
        self.layers = layers
        self.layers_by_depth = layers_by_depth
        
        #get sorted list of connection depths
        depth_keys = list(connections_by_depth.keys())
        depth_keys.sort(reverse=True)
        
        #check all required tensors are computable
        computable_tensors = []
        for x in self.inputs:
            computable_tensors.append(x)
        layers_with_complete_input = []
        for depth in depth_keys:
            for con in connections_by_depth[depth]:
                layer = con.outbound_model
                if(layer):
                    for x in con.input_tensors:
                        if(x not in computable_tensors):
                            raise Exception("Graph is disconnected")
                    for x in con.output_tensors:
                        computable_tensors.append(x)
                    layers_with_complete_input.append(layer.name)
                    
        #set self.connections and self.connections_by_depth
        self.container_connections = container_connections
        self.connections_by_depth = connections_by_depth
        
        #ensure name uniqueness. Crucial for serialisation
        all_names = [layer.name for layer in self.layers]
        for name in all_names:
            if(all_names.count(name) != 1):
                raise Exception("Layer names should be unique! Name reuse found.")
        
        self.outbound_connections = []
        self.inbound_connections = []
        MLConnection(self,
                     inbound_models=[], connection_indices=[],tensor_indices=[],
                    input_tensors=self.inputs, output_tensors=self.outputs,
                    input_shapes=[x._nuro_shape for x in self.inputs],
                    output_shapes=[x._nuro_shape for x in self.outputs])
        
        self.is_training = True
        self.is_built = True
        

        
    def run_internal_graph(self, inputs):
        """Computes output tensors for the graph
        given new input
        
        inputs: List of tensors (at least one)
        
        # Returns:
        output_tensors: List of outputs
        output_shapes: Shapes of output tensors
        """
        assert isinstance(inputs, list)
        
        tensor_map = {}
        for x, y in zip(self.inputs, inputs):
            tensor_map[str(id(x))] = y
            
        depth_keys = list(self.connections_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            connections = self.connections_by_depth[depth]
            for connection in connections:
                layer = connection.outbound_model
                ref_input_tensors = connection.input_tensors
                ref_output_tensors = connection.output_tensors
                
                #If all previous input tensors are available in
                #tensor_map, then call connection.inbound_model on them
                computed_data = []
                for x in ref_input_tensors:
                    if(str(id(x)) in tensor_map):
                        computed_data.append(tensor_map[str(id(x))])
                if(len(computed_data) == len(ref_input_tensors)):
                    #call layer
                    if(len(computed_data) == 1):
                        computed_tensor = computed_data[0]
                        output_tensors = make_list(layer.prop_up(computed_tensor))
                        computed_tensors = [computed_tensor]
                    else:
                        computed_tensors = [x[0] for x in computed_data]
                        output_tensors = make_list(layer.prop_up(computed_tensors))
                        
                    #Update _shape
                    if(all([hasattr(x, '_nuro_shape') for x in computed_tensors])):
                        shapes = make_list([layer.get_output_shape(x._nuro_shape) for x in computed_tensors])
                        for x, s in zip(output_tensors, shapes):
                            x._nuro_shape = s
                    
                    for x, y in zip(ref_output_tensors, output_tensors):
                        tensor_map[str(id(x))] = y
        
        
        output_tensors = []
        output_shapes = []
        for x in self.outputs:
            assert str(id(x)) in tensor_map, "Could not compute output " + str(x)
            tensor = tensor_map[str(id(x))]
            if(hasattr(tensor, '_nuro_shape') and output_shapes is not None):
                shape = tensor._nuro_shape
                output_shapes.append(shape)
            else:
                output_shapes = None
            output_tensors.append(tensor)
            
        cache_key = ','.join([str(id(i)) for i in inputs])
        
        if(len(output_tensors) == 1):
            output_tensors = output_tensors[0]
        self._output_tensor_cache[cache_key] = output_tensors
        
        if(output_shapes is not None):
            input_shapes = [x._nuro_shape for x in inputs]
            cache_key = ','.join([str(x) for x in input_shapes])
            if(len(output_shapes) == 1):
                output_shapes = output_shapes[0]
            self._output_shape_cache[cache_key] = output_shapes
        
        return output_tensors, output_shapes
        
        
    @property
    def trainable_weights(self):
        weights = []
        for layer in self.layers:
            if(layer.trainable):
                weights += layer.trainable_weights
        return weights
        
    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.layers:
            if(not layer.trainable):
                weights += layer.trainable_weights + layer.non_trainable_weights
        return weights
        
    def set_training(self, value):
        assert isinstance(value, bool), "set_training requires True/False arg"
        self.is_training = value
        for layer in self.layers:
            if(isinstance(layer, list)):
                [n.set_training(value) for n in layer]
            else:
                layer.set_training(value)
        
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return [N.get_value(x) for x in weights]
        
    def set_weights(self, weights):
        for layer in self.layers:
            n_params = len(layer.weights)
            layer_weights = weights[:n_params]
            for sw, w in zip(layer.weights, layer_weights):
                N.set_value(sw, w)
            weights = layer_weights[n_params:]
            
            
    def save_weights(self, filepath, overwrite=True):
        import h5py
        if(not overwrite and os.path.isfile(filepath)):
            proceed = can_save_with_overwrite(filepath)
            if(not proceed):
                return
        f = h5py.File(filepath, 'w')
        self.save_weights_to_hdf5_group(f)
        f.flush()
        f.close()
        
    def load_weights(self, filepath, by_name=False):
        """Loads all layer weights from a HDF5 save file.
        
        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.
        
        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.
        """
        import h5py
        f = h5py.File(filepath, mode='r')
        if('layer_names' not in f.attrs and 'model_weights' in f):
            f = f['model_weights']
        if(by_name):
            self.load_weights_from_hdf5_group_by_name(f)
        else:
            self.load_weights_from_hdf5_group(f)
        if(hasattr(f, 'close')):
            f.close()
            
    def save_weights_to_hdf5_group(self, f):
        f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in self.layers]
        for layer in self.layers:
            group = f.create_group(layer.name)
            symbolic_weights = layer.weights
            weight_vals = [N.get_value(x) for x in symbolic_weights]
            weight_names = []
            for i, w in enumerate(symbolic_weights):
                name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))
            group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_vals):
                param_dset = group.create_dataset(name, val.shape, dtype=val.dtype)
                if(not val.shape):
                    param_dset[()] = val
                else:
                    param_dset[:] = val
                    
    def load_weights_from_hdf5_group(self, f):
        filtered_layers = []
        for layer in self.layers:
            weights = layer.weights
            if(weights):
                filtered_layers.append(layer)
        
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names:
            group = f[name]
            weight_names = [n.decode('utf8') for n in group.attrs['weight_names']]
            if(len(weight_names)):
                filtered_layer_names.append(name)
        layer_names = filtered_layer_names
        if(len(layer_names) != len(filtered_layers)):
            raise Exception("Trying to load a file containing {} layers into a model with {} layers".format(len(layer_names), len(filtered_layers)))
            
        for i, name in enumerate(layer_names):
            group = f[name]
            weight_names = [n.decode('utf8') for n in group.attrs['weight_names']]
            weight_values = [group[weight_name] for weight_name in weight_names]
            layer = filtered_layers[i]
            symbolic_weights = layer.weights
            if(len(weight_values) != len(symbolic_weights)):
                raise Exception("Layer #"+str(i)+"has mismatching number of weights from the save file")
            for w, v in zip(symbolic_weights, weight_values):
                N.set_value(w, v)
                
    def load_weights_from_hdf5_group_by_name(self, f):
        """Name-base loading of weights.
        (Instead of usual topological weight loading)
        """
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        index = {}
        for layer in self.layers:
            if(layer.name):
                index.setdefault(layer.name, []).append(layer)
        
        for i, name in enumerate(layer_names):
            group = f[name]
            weight_names = [n.decode('utf8') for n in group.attrs['weight_names']]
            weight_values = [group[weight_name] for weight_name in weight_names]
            
            for layer in index.get(name, []):
                symbolic_weights = layer.weights
                if(len(weight_values) != len(symbolic_weights)):
                    raise Exception("Layer #"+str(i)+"has mismatching number of weights from the save file")
                
                for w, v in zip(symbolic_weights, weight_values):
                    N.set_value(w, v)
                    
            
    
            
            
    def get_layer(self, index=None):
        """Fetches a layer based on its index in the graph.
        Indices are based on bottom-up traversal
        """
        if(len(self.layers) <= index):
            raise Exception("get_layer was asked to fetch a layer with index "+ str(index) + \
                            ". But it only has " + str(len(self.layers)) + " layers.")
        else:
            return self.layers[index]
        
            
    def build(self):
        self.is_built = True
        
    def predict(self, *input_args):
        if(len(input_args) != len(self.inputs)):
            raise ValueError("Network model has "+str(len(self.inputs))+" inputs "
                             ".Only got {}".format(len(input_args)))
        if(not hasattr(self, '_predictor')):
            self._predictor = self.get_predictor()
        return self._predictor(*input_args)
        
    def get_predictor(self):
        training_mode = self.is_training
        self.set_training(False)
        function = N.function(self.inputs, self.prop_up(self.inputs))
        self.set_training(training_mode)
        return function
        
    def prop_up(self, x):
        x = make_list(x)
        cache_key = ','.join([str(id(i)) for i in x])
        if(cache_key in self._output_tensor_cache):
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, output_shapes = self.run_internal_graph(x)
            return output_tensors
            
    def get_updates(self):
        updates = []
        for layer in self.layers:
            new_updates = layer.get_updates()
            if(isinstance(new_updates, list)):
                updates += new_updates
            elif(isinstance(new_updates, OrderedDict)):
                updates += new_updates.items()
        return OrderedDict(updates)
        
    def get_cost(self):
        cost = None
        for layer in self.layers:
            if(cost is None):
                cost = layer.get_cost()
            else:
                cost += layer.get_cost()
        return cost
        
    def get_output_shape(self, input_shape):
        input_shapes = make_list(input_shape)
        if(len(input_shapes) != len(self.input_layers)):
            raise Exception("get_output_shape needs shapes for ALL its input layers")
            
        cache_key = ','.join([str(x) for x in input_shapes])
        if(cache_key in self._output_shape_cache):
            output_shapes = self._output_shape_cache[cache_key]
            if(isinstance(output_shapes, list) and len(output_shapes) == 1):
                return output_shapes[0]
            return output_shapes
        
        else:
            layerwise_output_shapes = {}
            for i in range(len(input_shapes)):
                layer = self.input_layers[i]
                input_shape = input_shapes[i]
                shape_key = layer.name + '_0_0'
                layerwise_output_shapes[shape_key] = input_shape
                
            depth_keys = list(self.connections_by_depth.keys())
            depth_keys.sort(reverse=True)
            
            if(len(depth_keys) > 1):
                for depth in depth_keys:
                    connections = self.connections_by_depth[depth]
                    for connection in connections:
                        layer = connection.outbound_model
                        if(layer in self.input_layers):
                            continue
                        input_shapes = []
                        for j in range(len(connection.inbound_models)):
                            inbound_model = connection.inbound_models[j]
                            connection_idx = connection.connection_indices[j]
                            tensor_idx = connection.tensor_indices[j]
                            shape_key = inbound_model.name + "_{}_{}".format(connection_idx, tensor_idx)
                            input_shape = layerwise_output_shapes[shape_key]
                            input_shapes.append(input_shape)
                            
                        if(len(input_shapes) == 1):
                            output_shape = layer.get_output_shape(input_shapes[0])
                        else:
                            output_shape = [layer.get_output_shape(in_shape) for in_shape in input_shapes]
                        output_shapes = make_list(output_shape)
                        connection_idx = layer.inbound_connections.index(connection)
                        for j in range(len(output_shapes)):
                            shape_key = layer.name + '_{}_{}'.format(connection_idx, j)
                            layerwise_output_shapes[shape_key] = output_shapes[j]
                        
            output_shapes = []
            output_shape_keys = []
            for i in range(len(self.output_layers)):
                layer = self.output_layers[i]
                connection_idx = self.output_layers_connection_indices[i]
                tensor_idx = self.output_layers_tensor_indices[i]
                shape_key = layer.name + '_{}_{}'.format(connection_idx, tensor_idx)
                output_shape_keys.append(shape_key)
                
            for i, key in enumerate(output_shape_keys):
                assert key in layerwise_output_shapes
                output_shapes.append(layerwise_output_shapes[key])
                
            #Store in cache
            self._output_shape_cache[cache_key] = output_shapes
            if(isinstance(output_shapes, list) and len(output_shapes) == 1):
                return output_shapes[0]
            return output_shapes
            
        
                
        

if __name__ == "__main__":
    from nuronet2.layers import Input, DenseLayer
    a = Input((2,))
    b = Input((4,))
    c = DenseLayer(5)

    tensor1 = c(a)
    tensor2 = c(b)
    
    net = NetworkModel([a, b], [tensor1, tensor2], 'fred')
    out = net([a, b])
    f = N.function([a, b], out)
        

    