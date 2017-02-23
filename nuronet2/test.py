import numpy
import os
from nuronet2.base import get_weightfactory, get_regulariser, MLModel
from nuronet2.activations import get_activation
from nuronet2.backend import N

def save(model, filepath, overwrite=True):
    
    import h5py
    if(not overwrite and os.path.isfile(filepath)):
        proceed = can_save_with_overwrite(filepath)
        if(not proceed):
            return
    
    f = h5py.File(filepath, 'w')
    
    weights_group = f.create_group('model_weights')
    model.save_weights_to_hdf5_group(weights_group)
    f.flush()
    f.close()
    