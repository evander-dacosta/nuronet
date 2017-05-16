# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:22:03 2017

@author: Evander
"""

import nuronet2 as nuro

fName = "/home/evander/Dropbox/data/animals/"
train_dir = fName + 'training_set'
test_dir = fName + 'test_set'

train_image_generator = nuro.ImageDataGenerator(rescale=1./255,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True)
test_image_generator = nuro.ImageDataGenerator(rescale=1./255)

training_set = train_image_generator.dataset_from_dir(train_dir,
                                                     target_size=(64, 64),
                                                     batch_size=32, 
                                                     class_mode='binary')
test_set = test_image_generator.dataset_from_dir(test_dir,
                                                 target_size=(64, 64),
                                                 batch_size=32, 
                                                 class_mode='binary')

model = nuro.NeuralNetwork()
model.add(nuro.Conv2dLayer((32, 3, 3), activation="tanh2", 
                           input_shape=(3, 64, 64)))
model.add(nuro.Maxpool2d(pool_size=(2, 2)))
model.add(nuro.Flatten())
model.add(nuro.DenseLayer(128, activation='tanh2'))
model.add(nuro.DenseLayer(1, activation='sigmoid'))

model.compile(nuro.Adam(lr=1e-4), 'binary_crossentropy', metrics=["acc"])
model.fit_generator(training_set, steps_per_epoch=100, 
                    n_epochs=25, validation_data=test_set, validation_steps=10)

        