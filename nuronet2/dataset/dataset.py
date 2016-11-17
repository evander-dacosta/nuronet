# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:14:13 2016

@author: Evander
"""

def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]

class Dataset(object):

    def __init__(self, batchSize):
        self.batchSize = batchSize

    def __call__(self, X, Y=None):
        return self.__iter__(X, Y)

    def __iter__(self, X, Y):
        bs = self.batchSize
        numSamples = self.get_dataset_size(X)
        for i in range(int((numSamples + bs - 1) // bs)):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = _sldict(X, sl)
            if(Y is not None):
                Yb = Y[sl]
            else:
                Yb = None
            yield self.transform(Xb, Yb)

    def get_dataset_size(self, X):
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    @property
    def num_samples(self):
        return self.get_dataset_size(self.X)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if(attr in state):
                del state[attr]
        return state