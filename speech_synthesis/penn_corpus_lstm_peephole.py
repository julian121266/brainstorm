#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import brainstorm as bs
from brainstorm.handlers import PyCudaHandler
from brainstorm.data_iterators import Minibatches, DataIterator
from brainstorm.initializers import Gaussian
from brainstorm.training.steppers import MomentumStep
from brainstorm.hooks import MonitorLoss
from brainstorm.value_modifiers import ClipValues
from brainstorm.tools import get_in_out_layers_for_classification
import numpy as np
import os
import sys
import math
import numpy as np
import sys
from brainstorm.randomness import Seedable
from brainstorm.utils import IteratorValidationError
from brainstorm.handlers._cpuop import _crop_images

if sys.version_info < (3,):
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

# ----------------------------- Iterator from index to one hot vector ------ #
class One_hot(DataIterator):
    """
    Data Iterator to map simple positional value of character in a vocabulary to one hot vectors.
    """
    def __init__(self, iter, vocab_size_dict):
        """

        """
        super(One_hot, self).__init__(iter.data_shapes, iter.length)
        if type(vocab_size_dict) is not dict:
            raise IteratorValidationError("Wrong data type for size of vocabulary. Should be dictionary.")
        self.vocab_size_dict = vocab_size_dict
        self.iter = iter

    def __call__(self, handler):
        for data in self.iter(handler):
            for name in self.vocab_size_dict.keys():
                new_data = np.eye(self.vocab_size_dict[name], dtype=np.bool)[data[name]]
                new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[1], new_data.shape[3]))
                data[name] = new_data
                yield data

# ------------------------------ Get the data ------------------------------- #

def preprocess_data(datafilepath):
    datafile = open(datafilepath, 'r').read()
    data_new = datafile.replace(' ', '')
    data_new = data_new.replace('\\/', '')
    # data_new = data_new.replace('\n', '')
    # print(data_new)
    return data_new

def prepare_data(data_new, vocab_dict, sequence_length=100):
    # print('')
    # print('Sample text:', data_new[0:200])
    data = np.array([vocab_dict[char] for char in data_new], dtype=int)
    reverse_dict = dict([(v, k) for (k, v) in vocab_dict.items()])
    data_reverse = [reverse_dict[char] for char in data]  # for testing purposes
    data_mask = np.ones(data.shape)
    data_targets = data[1:]  # one step ahead
    data = data[0:-1]  # make same size as targets (last data point cannot predict anything)
    data_mask = data_mask[0:-1]
    N_batch = data.shape[0]//sequence_length

    new_data = np.zeros((sequence_length, N_batch, 1), dtype=int)
    new_target = np.zeros((sequence_length, N_batch, 1), dtype=int)
    new_mask = np.zeros((sequence_length, N_batch, 1), dtype=int)

    for i in range(N_batch):
        for j in range(sequence_length):
            new_data[j, i, 0] = data[j+i*sequence_length]
            new_mask[j, i, 0] = data_mask[j+i*sequence_length]
            new_target[j, i, 0] = data_targets[j+i*sequence_length]
    data = new_data
    # data_reverse = [reverse_dict[char] for char in np.squeeze(np.append(data[:, 0], data[:, 1]))]
    # print('Reversely mapped sample text', ''.join(data_reverse[0:200]))
    # print('')
    data_mask = new_mask
    data_targets = new_target

    assert(data.shape == data_targets.shape)
    assert(data.shape == data_mask.shape)
    assert(len(data.shape) == 3)
    assert(len(data_targets.shape) == 3)
    assert(len(data_mask.shape) == 3)

    return data, data_targets, data_mask


def construct_period_mask(periods, configuration='S2F'):
    """
    Construct a mask for the recurrent matrix of an Clockwork_RNN/LSTM layer, to ensure that
    connections only go to units of higher frequency, but not back or vice versa.
    """
    assert(configuration == 'S2F' or configuration == 'F2S'), \
        "Please choose correct configuration: S2F (slow2fast) or F2S (fast2slow)"
    unique_ps = sorted(set(periods))
    connection_matrix = np.zeros((len(periods), len(periods)), dtype=np.float64)
    offset = 0
    if configuration == 'S2F':
        for p in unique_ps:
            group_size = unique_ps.count(p)
            connection_matrix[offset:, offset:offset + group_size] = 1.0
            offset += group_size
    elif configuration == 'F2S':
        for p in unique_ps:
            group_size = unique_ps.count(p)
            connection_matrix[offset:offset + group_size, offset:] = 1.0
            offset += group_size
    return connection_matrix


def PrepareClockwork(size_cw_net_per_group, n_groups):
    size_cw = size_cw_net_per_group*n_groups
    timing = 1 * np.ones(size_cw_net_per_group, dtype=int)
    for i in range(1, n_groups):
        timing = np.concatenate((timing, pow(2, i) * np.ones(size_cw_net_per_group, dtype=int)), axis=0)
    return size_cw, timing

def RandInitMatrix(size):
    return np.random.randn(size, size)

# ----------------------------- Set up Data ------------------------------ #

train = 'data/ptb.char.train.txt'
valid = 'data/ptb.char.valid.txt'
test = 'data/ptb.char.test.txt'

train = preprocess_data(train)
valid = preprocess_data(valid)
test = preprocess_data(test)

vocab = np.unique(list(train)+list(valid)+list(test))
vocab_dict = dict(zip(list(vocab), range(0, len(vocab))))

print('Length of Dictionary:', len(vocab_dict))
print('Dictionary:', vocab_dict)
sequence_length = 100  # length of time sequence into which the symbols are rearranged

train_inputs, train_targets, train_mask = prepare_data(train, vocab_dict, sequence_length)
valid_inputs, valid_targets, valid_mask = prepare_data(valid, vocab_dict, sequence_length)
test_inputs, test_targets, test_mask = prepare_data(test, vocab_dict, sequence_length)

# ----------------------------- Set up Network ------------------------------ #
n_classes = len(vocab_dict)
configuration = 'S2F'
size_cw, timing = PrepareClockwork(166, 6)
config_mask = construct_period_mask(timing, configuration='S2F')

inp, out = bs.tools.get_in_out_layers_for_classification(n_classes, n_classes, outlayer_name='out',
                                                   mask_name='mask')
# inp >> bs.layers.LstmPeephole(1000, name='lstm_peep') >> out
inp >> bs.layers.ClockworkLstmPeep(size_cw, timing, name='cw_lstm_peep') >> out
network = bs.Network.from_layer(out)
# network = bs.Network.from_hdf5('penn_corpus_best_lstm_peep_batchsize1.hdf5')

network.set_handler(PyCudaHandler())
# network.initialize({"default": bs.initializers.Gaussian(0.1)}, seed=42)
network.initialize({"default": bs.initializers.Gaussian(0.1), "cw_lstm_peep":\
    {'Rz': RandInitMatrix(len(timing))*config_mask, 'Ri': RandInitMatrix(len(timing))*config_mask,\
     'Rf': RandInitMatrix(len(timing))*config_mask, 'Ro': RandInitMatrix(len(timing))*config_mask}}, seed=42)

network.set_weight_modifiers({"cw_lstm_peep": {'Rz': bs.value_modifiers.MaskValues(config_mask),\
                                           'Ri': bs.value_modifiers.MaskValues(config_mask),\
                                           'Rf': bs.value_modifiers.MaskValues(config_mask),\
                                           'Ro': bs.value_modifiers.MaskValues(config_mask)}})

network.set_gradient_modifiers({"cw_lstm_peep": bs.value_modifiers.ClipValues(low=-1., high=1)})

# ---------------------------- Set up Iterators ----------------------------- #
train_getter = bs.data_iterators.Minibatches(100, False, mask=train_mask,  # WITH OR WITHOUT SHUFFLING?
                              default=train_inputs, targets=train_targets)
valid_getter = bs.data_iterators.Minibatches(100, False, mask=valid_mask,
                              default=valid_inputs, targets=valid_targets)
test_getter = bs.data_iterators.Minibatches(100, False, mask=test_mask,
                             default=test_inputs, targets=test_targets)

vocab_dict_name = {'default': len(vocab_dict)}
train_getter = One_hot(train_getter, vocab_dict_name)
valid_getter = One_hot(valid_getter, vocab_dict_name)
test_getter = One_hot(test_getter, vocab_dict_name)

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.training.MomentumStep(learning_rate=0.01, momentum=0.99),
                     double_buffering=False)
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='out.probabilities')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation'))
trainer.add_hook(bs.hooks.SaveBestNetwork('validation.Accuracy',
                                          filename='penn_corpus_best_cw_lstm1.hdf5',
                                          name='best weights',
                                          criterion='max'))
#  penn_corpus_best_lstm_peeplong
trainer.add_hook(bs.hooks.StopAfterEpoch(500))

# -------------------------------- Train ------------------------------------ #
trainer.train(network, train_getter, valid_getter=valid_getter)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))