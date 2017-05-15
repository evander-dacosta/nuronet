# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:30:17 2017

@author: Evander
"""
from nuronet2.backend import N

def binary_accuracy(y_true, y_pred):
    return N.mean(N.equal(y_true, N.round(y_pred)))


def categorical_accuracy(y_true, y_pred):
    return N.cast(N.equal(N.argmax(y_true, axis=-1),
                          N.argmax(y_pred, axis=-1)),
                  N.floatx)