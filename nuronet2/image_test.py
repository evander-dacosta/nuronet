# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:22:03 2017

@author: Evander
"""

import nuronet2 as nuro
import numpy

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
p = 0.5
model = nuro.NeuralNetwork()
# Layer 1
model.add(nuro.Conv2dLayer((32, 3, 3), activation="linear", 
                           input_shape=(3, 64, 64)))
model.add(nuro.BatchNormalisation(axis=1))
model.add(nuro.Activation("relu"))
model.add(nuro.Maxpool2d(pool_size=(2, 2)))
model.add(nuro.BatchNormalisation(axis=1))

# Layer 2
model.add(nuro.Conv2dLayer((32, 3, 3), activation="linear"))
model.add(nuro.BatchNormalisation(axis=1))
model.add(nuro.Activation("relu"))
model.add(nuro.Maxpool2d(pool_size=(2, 2)))
model.add(nuro.BatchNormalisation(axis=1))

# Layer 3
model.add(nuro.Conv2dLayer((64, 3, 3), activation="linear"))
model.add(nuro.BatchNormalisation(axis=1))
model.add(nuro.Activation("relu"))
model.add(nuro.Maxpool2d(pool_size=(2, 2)))
model.add(nuro.BatchNormalisation(axis=1))

#Layer 4
model.add(nuro.Conv2dLayer((64, 3, 3), activation="linear"))
model.add(nuro.BatchNormalisation(axis=1))
model.add(nuro.Activation("relu"))
model.add(nuro.Maxpool2d(pool_size=(2, 2)))
model.add(nuro.BatchNormalisation(axis=1))

model.add(nuro.Flatten())
model.add(nuro.DenseLayer(256, activation="linear"))
model.add(nuro.BatchNormalisation(axis=1))
model.add(nuro.Activation("relu"))
model.add(nuro.Dropout(p))
model.add(nuro.DenseLayer(128, activation="linear"))
model.add(nuro.BatchNormalisation(axis=1))
model.add(nuro.Activation("relu"))
model.add(nuro.DenseLayer(64, activation="linear"))
model.add(nuro.BatchNormalisation(axis=1))
model.add(nuro.Activation("relu"))
model.add(nuro.Dropout(p/2))
model.add(nuro.DenseLayer(1, activation='sigmoid'))

model.compile(nuro.Adam(lr=1e-3), 'binary_crossentropy', metrics=["acc"])
model.fit_generator(training_set, steps_per_epoch=250, 
                    n_epochs=100, validation_data=test_set, 
                    validation_steps=20)

def get_accuracy(x_t, y_t):
    pred = model.predict(x_t)
    pred = pred>0.5
    assert(pred.shape == y_t.shape)
    return numpy.sum(pred == y_t)
    
count = 0
samples = 0
for i in range(63):
    x_t, y_t = test_set.next()
    samples += x_t.shape[0]
    count += get_accuracy(x_t, y_t)
print "Training accuracy is {}%".format(count*100./float(samples))