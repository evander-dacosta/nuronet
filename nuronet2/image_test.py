# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:22:03 2017

@author: Evander
"""

import nuronet2 as nuro


fName = "/home/evander/Dropbox/data/animals/"
#fName = "C:\\Users\\Evander\\Dropbox\\data\\animals\\"

train_dir = fName + 'training_set'
test_dir = fName + 'test_set'

train_image_generator = nuro.ImageDataGenerator(rescale=1./255,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True)
test_image_generator = nuro.ImageDataGenerator(rescale=1./255)

training_set = train_image_generator.dataset_from_dir(train_dir,
                                                     target_size=(64, 64),
                                                     batch_size=32, class_mode='binary')
test_set = test_image_generator.dataset_from_dir(test_dir,
                                                 target_size=(64, 64),
                                                 batch_size=32, class_mode='binary')
            

                    
model = nuro.NeuralNetwork()
model.add(nuro.Conv2dLayer((32, 3, 3), activation="relu", input_shape=(3, 64, 64)))
model.add(nuro.Maxpool2d(pool_size=(2, 2), strides=(1, 1)))
model.add(nuro.Flatten())
model.add(nuro.DenseLayer(128, activation='relu'))
model.add(nuro.DenseLayer(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy')
model.fit_generator(training_set, n_epochs=50)
