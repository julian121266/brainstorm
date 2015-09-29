#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
from brainstorm.structure.construction import ConstructionWrapper
from brainstorm.utils import LayerValidationError, flatten_time
from brainstorm.layers.base_layer import LayerBaseImpl
from brainstorm.structure.shapes import ShapeTemplate

def ClockworkRnn(size, timing, activation_function='tanh', name=None):
    return ConstructionWrapper.create('ClockworkRnn',
                                      size=size,
                                      timing=timing,
                                      name=name,
                                      activation_function=activation_function)


class ClockworkRnnLayerImpl(LayerBaseImpl):
    expected_kwargs = {'size', 'timing', 'activation_function'}

    def _setup_hyperparameters(self):
        self.act_func = None
        self.act_func_deriv = None
        self.size = self.kwargs.get('size',
                                    self.in_shapes['default'].feature_size)

        if not isinstance(self.size, int):
            raise LayerValidationError('size must be int but was {}'.
                                       format(self.size))

    def set_handler(self, new_handler):
        super(ClockworkRnnLayerImpl, self).set_handler(new_handler)

        # Assign act_func and act_dunc_derivs
        activation_functions = {
            'sigmoid': (self.handler.sigmoid, self.handler.sigmoid_deriv),
            'tanh': (self.handler.tanh, self.handler.tanh_deriv),
            'linear': (lambda x, y: self.handler.copy_to(y, x),
                       lambda x, y, dy, dx: self.handler.copy_to(dx, dy)),
            'rel': (self.handler.rel, self.handler.rel_deriv)
        }

        self.act_func, self.act_func_deriv = activation_functions[
            self.kwargs.get('activation_function', 'tanh')]

    def get_parameter_structure(self):
        in_size = self.in_shapes['default'].feature_size
        parameters = OrderedDict()
        parameters['W'] = ShapeTemplate(self.size, in_size)  #  size now refers to size within each group
        parameters['R'] = ShapeTemplate(self.size, self.size)  #  size now refers to size within each group
        parameters['bias'] = ShapeTemplate(self.size)

        parameters['timing'] = ShapeTemplate(self.size)

        return parameters

    def get_internal_structure(self):
        internals = OrderedDict()
        internals['Ha'] = ShapeTemplate('T', 'B', self.size, context_size=1)
        internals['dHa'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        internals['dHb'] = ShapeTemplate('T', 'B', self.size, context_size=1,
                                         is_backward_only=True)
        return internals

    def _get_output_shapes(self):
        return {'default': ShapeTemplate('T', 'B', self.size, context_size=1)}


    def forward_pass(self, buffers, training_pass=True):
        # prepare
        _h = self.handler
        W, R, bias, timing = buffers.parameters
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        Ha = buffers.internals.Ha

        batch_size = W.shape[1]
        feature_size = timing.shape[0]

        flat_inputs = flatten_time(inputs)
        flat_H = flatten_time(Ha[:-1])
        _h.dot_mm(flat_inputs, W, flat_H, transb=True)
        _h.add_mv(flat_H, bias.reshape((1, self.size)), flat_H)
        # do normal forward pass, then set all hidden unit activations with t!=T to the previous activation
        # (or more briefly said: do not update hidden units that are not active)

        tmp = _h.zeros(timing.shape)
        # tmp2 = _h.zeros(timing.shape)
        for t in range(inputs.shape[0]):
            _h.dot_add_mm(outputs[t - 1], R, Ha[t])
            self.act_func(Ha[t], outputs[t])
            # NEW -------------------------------------------------------------------------------------
            if t > 0:
                _h.fill(tmp, t)
                _h.modulo_mm(tmp, timing, tmp)

                # For inactive nodes activations are reset
                _h.clw_undo_update(batch_size, feature_size, tmp, outputs[t-1], outputs[t])
                # _h.clw_undo_update(batch_size, feature_size, tmp, Ha[t-1], Ha[t])

            # NEW END ---------------------------------------------------------------------------------

    def backward_pass(self, buffers):
        # prepare
        _h = self.handler
        W, R, bias, timing = buffers.parameters
        dW, dR, dbias, dTiming = buffers.gradients
        inputs = buffers.inputs.default
        outputs = buffers.outputs.default
        dinputs = buffers.input_deltas.default
        doutputs = buffers.output_deltas.default
        Ha, dHa, dHb = buffers.internals

        tmp = _h.zeros(timing.shape)
        batch_size = W.shape[1]
        feature_size = timing.shape[0]


        _h.copy_to(dHb, doutputs)
        T = inputs.shape[0] - 1
        self.act_func_deriv(Ha[T], outputs[T], dHb[T], dHa[T])
        for t in range(T - 1, -1, -1):

            _h.fill(tmp, t+1)
            _h.modulo_mm(tmp, timing, tmp)
            _h.clw_copy_add_act_of_inactive(batch_size, feature_size, tmp, dHb[t+1], dHb[t])
            _h.clw_set_inactive_to_zero(batch_size, feature_size, tmp, dHa[t+1])

            _h.dot_add_mm(dHa[t + 1], R, dHb[t], transb=True)
            self.act_func_deriv(Ha[t], outputs[t],
                                dHb[t], dHa[t])

            # set inactive nodes to 0 (activation delta)
            # _h.fill(tmp, t)
            # _h.modulo_mm(tmp, timing, tmp)


        # Same as for standard RNN:
        flat_inputs = flatten_time(inputs)
        flat_dinputs = flatten_time(dinputs)
        flat_dHa = flatten_time(dHa[:-1])

        # calculate in_deltas and gradients
        _h.dot_add_mm(flat_dHa, W, flat_dinputs)
        _h.dot_add_mm(flat_dHa, flat_inputs, dW, transa=True)
        dbias_tmp = _h.allocate(dbias.shape)
        _h.sum_t(flat_dHa, axis=0, out=dbias_tmp)
        _h.add_tt(dbias, dbias_tmp, dbias)

        flat_outputs = flatten_time(outputs[:-2])
        flat_dHa = flatten_time(dHa[1:-1])
        _h.dot_add_mm(flat_outputs, flat_dHa, dR, transa=True)
        _h.dot_add_mm(outputs[-1], dHa[0], dR, transa=True)
