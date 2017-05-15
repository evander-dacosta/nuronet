"""
Created on Mon May 15 14:58:58 2017

@author: Evander
"""

import os
import numpy
import warnings
import scipy.ndimage as ndi
import itertools
from sklearn.cross_validation import KFold

from nuronet2.dataset import IndexIterator
from nuronet2.backend import N

import matplotlib.pyplot as plt
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None



def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = numpy.pi / 180 * numpy.random.uniform(-rg, rg)
    rotation_matrix = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                                [numpy.sin(theta), numpy.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = numpy.random.uniform(-hrg, hrg) * h
    ty = numpy.random.uniform(-wrg, wrg) * w
    translation_matrix = numpy.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
    shear = numpy.random.uniform(-intensity, intensity)
    shear_matrix = numpy.array([[1, -numpy.sin(shear), 0],
                             [0, numpy.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = numpy.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = numpy.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = numpy.rollaxis(x, channel_axis, 0)
    min_x, max_x = numpy.min(x), numpy.max(x)
    channel_images = [numpy.clip(x_channel + numpy.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = numpy.stack(channel_images, axis=0)
    x = numpy.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = numpy.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = numpy.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = numpy.dot(numpy.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = numpy.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = numpy.stack(channel_images, axis=0)
    x = numpy.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = numpy.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def img_to_array(img):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = numpy.asarray(img, dtype=N.floatx)
    if len(x.shape) == 3:
        x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        x = x.reshape((1, x.shape[0], x.shape[1]))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x
    
def plot_img(img):
    if(len(img.shape) == 3):
        img = img.transpose(1, 2, 0)
    plt.imshow(img)


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img
    
    
class ImageDataGenerator(object):
    """
    Generates minibatches of image data with real-time
    data augmentation
    
    # Arguments
    featurewise_center: set input means to 0 over dataset
    
    samplewise_center: set each sample mean to 0
    
    featurewise_std_normalisation: divide inputs by std of the dataset
    
    samplewise_std_normalisation: divide each input by its std
    
    zca_whitening: apply ZCA whitening
    
    rotation_range: degrees (0 to 180)
    
    width_shift_range: fraction of total width
    
    height_shift_range: fraction fo total height
    
    shear_range: shear intensity (angle in radians)
    
    zoom_range: amount of zoom. If scalar a, the zoom
                will be randomly picked between [1-z, 1+z].
                A sequence of the two can be passed instead, to 
                select the range
    
    channel_shift_range: shift range for each channel
    
    fill_mode: Points outside the boundaries are filled according 
               to the given mode ('constant', 'nearest', 'reflect', or 'wrap')
               Default is nearest.
               
               
    cval: value used for the points outside the boundaries when fill_mode is 
          constant. Default is 0
          
    horizontal_flip: True/False
    
    vertical_flip: True/False randomly flip images vertically
    
    rescale: Rescaling factor. If None or 0, no rescaling is applied. 
             Otherwise all images are multiplied by this number (before applying
             any other transformation)
             
    preprocessing_function: Function that will be applied to each input
                            The function should take one input:
                            An image with tensor rank 3
                            and return a numpy tensor with same shape
                            
    """
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalisation=False,
                 samplewise_std_normalisation=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 ):
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalisation = featurewise_std_normalisation
        self.samplewise_std_normalisation = samplewise_std_normalisation
        self.zca_whitening = zca_whitening
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        
        self.channel_axis = 1
        self.row_axis = 2
        self.col_axis = 3
        
        self.mean = None
        self.std = None
        self.principal_components = None
        
        if(numpy.isscalar(zoom_range)):
            self.zoom_range = [1-zoom_range, 1+zoom_range]
        elif(len(zoom_range) == 2):
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('zoom_range must be scalar '
                             'or a list/tuple of length 2')
        
        self.preprocessing_function = preprocessing_function
    
    def dataset_from_dir(self, directory,
                         target_size=(256, 256), colour_mode='rgb',
                         classes=None, class_mode='categorical', 
                         batch_size=32, shuffle=True,
                         follow_links=False):
        return DirectoryIterator(directory, self,
                                 target_size=target_size, colour_mode=colour_mode,
                                 classes=classes, class_mode=class_mode,
                                 batch_size=batch_size, shuffle=shuffle, 
                                 follow_links=follow_links)
                                 
    def standardise(self, x):
        """
        Apply normalisation to a batch of inputs x
        
        # Arguments
            x : Batch of inputs to be normalized
            
        """
        if(self.preprocessing_function):
            x = self.preprocessing_function(x)
        if(self.rescale):
            x *= self.rescale
            
        # x is a single image so it doesn't have
        # an index number at index 0
        img_channel_axis = self.channel_axis - 1
        if(self.samplewise_center):
            x -= numpy.mean(x, axis=img_channel_axis, keepdims=True)
        if(self.samplewise_std_normalisation):
            x /= (numpy.stff(x, axis=img_channel_axis, keepdims=True) + 1e-7)
        
        if(self.featurewise_center):
            if(self.mean is not None):
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                '"featurewise_center", but it hasnt been fit '
                'on any training data. Fit it by first calling .fit(numpy_data)')
                
        if(self.featurewise_std_normalisation):
            if(self.std is not None):
                x /= (self.std + 1e-7)
                
            else:
                warnings.warn('This ImageDataGenerator specifies '
                'featurewise_std_normalisation, but has not been fit on any '
                'training data. Fit it by first calling .fit(numpy_data)')
                
        return x
        
    def random_transform(self, x):
        """
        Randomly augment a single image tensor
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1
        
        if(self.rotation_range):
            theta = numpy.pi / (180. * numpy.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0.
        
        if self.height_shift_range:
            tx = numpy.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = numpy.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = numpy.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = numpy.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = numpy.array([[numpy.cos(theta), -numpy.sin(theta), 0],
                                        [numpy.sin(theta), numpy.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = numpy.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else numpy.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = numpy.array([[1, -numpy.sin(shear), 0],
                                    [0, numpy.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else numpy.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = numpy.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else numpy.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if numpy.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if numpy.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        return x
        
        
    def fit(self, x, augment=False, rounds=1,
            seed=None):
        """
        Fits internal statistics to some sample data.
        
        Required for featurewise_center, featurewise_std_normalisation,
                     samplewise_center, samplewise_std_normalisation
                     and zca whitening
        
        # Arguments
            x: numpy array. Should have rank 4 (batch_size, channel, width, height)
            
            augment: Whether to fit on randomly augmented samples
            
            rounds: If 'augment', how many augmentation passes to do over data
            
            seed: random seed
        """
        x = numpy.asarray(x, dtype=N.floatx)
        if(x.ndim != 4):
            raise ValueError("Must provide numpy tensor of rank 4")
        
        if(seed is not None):
            numpy.random.seed(seed)
        
        x = numpy.copy(x)
        if(augment):
            ax = numpy.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=N.floatx)
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax
            
        if self.featurewise_center:
            self.mean = numpy.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = numpy.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = numpy.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = numpy.reshape(self.std, broadcast_shape)
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flat_x = numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = numpy.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = numpy.linalg.svd(sigma)
            self.principal_components = numpy.dot(numpy.dot(u, numpy.diag(1. / numpy.sqrt(s + 10e-7))), u.T)


class DirectoryIterator(IndexIterator):
    """
    This iterator is used to read images from a
    folder on disk
    
    # Arguments
        directory: Path to read from

        image_data_generator: Instance of 'ImageDataGenerator'

        target_size: Tuple of integers, dimensions to resize images to.
        
        colour_mode: 'rgb' or 'grayscale'

        classes: Optional list of strings, names of subdirectories
                 containing names of each class e.g. ['dogs', 'cats']
                 
        class_mode: Method for yielding targets:
                    'binary': For binary 1/0 targets
                    'categorical': For categorical targets
                    None: no targets. Only images will be yielded
        
        batch_size: batch size
        
        validation: Fraction of dataset to use as a validation set
        
        shuffle: Whether or not to shuffle data between epochs
    """
    
    def __init__(self, directory, image_data_generator, target_size=(256, 256),
                 colour_mode='rgb', classes=None, class_mode=None,
                 batch_size=32, validation=0., shuffle=True, follow_links=False):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if(colour_mode not in ['rgb', 'grayscale']):
            raise ValueError('Invalid Colour Mode: ' , colour_mode,
                             '; Expected "rgb" or "grayscale".')
        
        self.colour_mode = colour_mode
        if(self.colour_mode == 'rgb'):
            self.image_shape = (3, ) + self.target_size
        else:
            self.image_shape = (1, ) + self.target_size
            
        self.classes = classes
        
        if(class_mode not in ['categorical', 'binary', 'None', None]):
            raise ValueError('Unknown class_mode')
            
        self.class_mode = class_mode
        
        list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        
        # first, count the number of samples and classes
        self.samples = 0
        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if(os.path.isdir(os.path.join(directory, subdir))):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])
            
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in list_formats:
                        if(fname.lower().endswith('.'+extension)):
                            is_valid = True
                            break
                    if(is_valid):
                        self.samples += 1
        print "Found {} images belonging to {} classes".format(self.samples, self.num_class)
        


        # Next, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = numpy.zeros((self.samples, ), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in list_formats:
                        if(fname.lower().endswith('.'+extension)):
                            is_valid = True
                            break
                    if(is_valid):
                        self.classes[i] = self.class_indices[subdir]
                        i += 1
                        absolute_path = os.path.join(root, fname)
                        self.filenames.append(os.path.relpath(absolute_path, directory))
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle)

        self.valid_indices = []        
        #number of validation samples
        if(validation is not None and validation > 0.):
            self.validation = validation
            self.valid_iterator = self.make_validation_iterator(self.samples)

    @property
    def x_valid(self):
        return self.x[self.valid_indices]
        
    @property
    def y_valid(self):
        return self.y[self.valid_indices]

    def make_validation_iterator(self, n):
        n_folds = round(1. / self.validation)
        kfold = KFold(n, n_folds)
        return itertools.cycle(kfold)

    def make_validation_splits(self):
        if(not hasattr(self, 'validation')):
            return
        _, valid_indices = next(self.valid_iterator)
        self.valid_indices = valid_indices
        self.reset()
            

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = numpy.zeros((current_batch_size, ) + self.image_shape, dtype=N.floatx)
        grayscale = self.colour_mode == 'grayscale'
        
        # Build a batch of image data
        for i, j in enumerate(index_array):
            f_name = self.filenames[j]
            img = load_img(os.path.join(self.directory, f_name),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardise(x)
            batch_x[i] = x
            
        if(self.class_mode == None or self.class_mode == "None"):
            batch_y = batch_x.copy()
        elif(self.class_mode == 'binary'):
            batch_y = self.classes[index_array].astype(N.floatx)
            batch_y = batch_y.reshape((batch_y.shape[0], 1))
        elif(self.class_mode == 'categorical'):
            batch_y = numpy.zeros((len(batch_x), self.num_class), dtype=N.floatx)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        

        return batch_x, batch_y

if __name__ == "__main__":
    #directory = "/home/evander/Dropbox/data/animals/training_set"
    directory = "C:\\Users\\Evander\\Dropbox\\data\\animals\\training_set"
    train_image_generator = ImageDataGenerator(rescale=1./255,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True)
    training_set = train_image_generator.dataset_from_dir(directory,
                                                     target_size=(64, 64),
                                                     batch_size=32, class_mode='binary')
    
    import time 
    def time_it():
        start_time = time.time()
        sho = 0
        for i in range(250):
            print sho
            x, y = training_set.next()
            sho += x.shape[0]
        end_time = time.time() - start_time
        return end_time
    print time_it()