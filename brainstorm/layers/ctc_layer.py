#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np

from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)
neg_inf = float('-inf')

def Ctc(name=None):
    return ConstructionWrapper.create('Ctc', name=name)

def calculate_alphas(Y_log, T):
    """
    Y_log: log of outputs shape=(time, labels)
    T: target sequence shape=(length, )
    """
    N = Y_log.shape[0]
    S = len(T)
    Z = 2 * S + 1

    alpha = np.full((N,Z), neg_inf)
    alpha[0, 0] = Y_log[0, 0]
    alpha[0, 1] = Y_log[0, T[0]]
    for t in range(1, N):
        start = max(-1, 2 * (S - N + t) + 1)
        for s in range(start + 1, Z, 2): #loop for blanks (even)
            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s])
            if s > 0:
                alpha[t,s] = np.logaddexp(alpha[t, s], alpha[t - 1, s - 1])
                
            alpha[t, s] += Y_log[t, 0]
        previous_label = -1
        if start > 0:
            previous_label = T[start // 2 - 1]
        for s in range(max(1, start), Z, 2): # loop for labels (odd)
            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s])
            alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s - 1])
            label = T[s // 2]
            if s > 1:
                alpha[t, s] = np.logaddexp(alpha[t, s], alpha[t - 1, s - 2] + np.log(label != previous_label))
            alpha[t, s] += Y_log[t, label]
            previous_label = label

    return alpha

class CtcLayerImpl(BaseLayerImpl):
    expected_kwargs = {}
    expected_inputs = {'default': StructureTemplate('T', 'B', 'F'),
                       'labels': StructureTemplate('T', 'B', 1)}

    def setup(self, kwargs, in_shapes):
        outputs = OrderedDict()
        outputs['default'] = BufferStructure('B', 1)
        parameters = OrderedDict()
        internals = OrderedDict()

        return outputs, parameters, internals

    def forward_pass(self, buffers, training_pass=True):
        pass

    def backward_pass(self, buffers):
        pass
