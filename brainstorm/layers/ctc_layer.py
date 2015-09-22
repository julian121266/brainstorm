#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals


from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.structure.shapes import ShapeTemplate


def Ctc(name=None):
    return ConstructionWrapper.create('Ctc', name=name)


class CtcLayerImpl(LayerBaseImpl):
    inputs = {'default': ShapeTemplate('T', 'B', 'F'),
              'labels': ShapeTemplate('T', 'B', 1)}

    outputs = {'loss': ShapeTemplate('B', 1)}

    def _get_output_shapes(self):
        return {'loss': ShapeTemplate('B', 1)}

    def forward_pass(self, buffers, training_pass=True):
        pass

    def backward_pass(self, buffers):
        pass
