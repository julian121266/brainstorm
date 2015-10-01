#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict

from brainstorm.layers.base_layer import BaseLayerImpl
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.structure.buffer_structure import (BufferStructure,
                                                   StructureTemplate)


def Ctc(name=None):
    return ConstructionWrapper.create('Ctc', name=name)


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
