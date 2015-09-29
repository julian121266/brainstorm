#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import brainstorm as bs
from brainstorm.handlers import PyCudaHandler
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


class DataIterator(object):
    def __init__(self, data):
        """
        BaseClass for Data Iterators.
        :param data: Named data items to iterate over.
        :type data: dict[unicode, np.ndarray]
        """
        self.data = data

    def __call__(self, handler, verbose=False):
        pass
# ----------------------------- Iterator from index to one hot vector ------ #
class One_hot(DataIterator):

    """
    Data Iterator to map simple positional value of character in a vocabulary to one hot vectors.
    """

    def __init__(self, iter, vocab_size_dict):
        """

        """
        DataIterator.__init__(self, iter.data)
        if type(vocab_size_dict) is not dict:
            raise IteratorValidationError("Wrong data type for size of vocabulary. Should be dictionary.")
        self.vocab_size_dict = vocab_size_dict
        self.iter = iter

    def __call__(self, handler, verbose=False):
        # for data in self.iter(handler):
        for data in self.iter(handler, verbose=verbose):
            for name in self.vocab_size_dict.keys():
                new_data = np.eye(self.vocab_size_dict[name], dtype=np.bool)[data[name]]
                new_data = np.squeeze(new_data)
                data[name] = new_data
                yield data

# ------------------------------ Get the data ------------------------------- #

def preprocess_data(datafilepath):
    datafile = open(datafilepath, 'r').read()
    data_new = datafile.replace(' ', '')
    data_new = data_new.replace('\\', '')
    return data_new

train = '../data/data/ptb.char.train.txt'
valid = '../data/data/ptb.char.valid.txt'
test = '../data/data/ptb.char.test.txt'

train = preprocess_data(train)
valid = preprocess_data(valid)
test = preprocess_data(test)

vocab = np.unique(list(train)+list(valid)+list(test))
vocab_dict = dict(zip(list(vocab), range(0, len(vocab))))

print('Length of Dictionary:', len(vocab_dict))
print('Dictionary:', vocab_dict)


size_batch = 100

def prepare_data(data_new, vocab_dict, size_batch=100):
    print('Sample text:', data_new[0:200])
    data = np.array([vocab_dict[char] for char in data_new], dtype=int)
    reverse_dict = dict([(v, k) for (k, v) in vocab_dict.items()])
    data_reverse = [reverse_dict[char] for char in data]  # for testing purposes
    data_mask = np.ones(data.shape)
    data_targets = data[1:]  # one step ahead
    data = data[0:-1]  # make same size as targets (last data point cannot predict anything)
    data_mask = data_mask[0:-1]
    N_batch = data.shape[0]//size_batch

    new_data = np.zeros((size_batch, N_batch, 1), dtype=int)
    new_target = np.zeros((size_batch, N_batch, 1), dtype=int)
    new_mask = np.zeros((size_batch, N_batch, 1), dtype=int)

    for i in range(N_batch):
        for j in range(size_batch):
            new_data[j, i, 0] = data[j+i*size_batch]
            new_mask[j, i, 0] = data_mask[j+i*size_batch]
            new_target[j, i, 0] = data_targets[j+i*size_batch]
    data = new_data
    data_reverse = [reverse_dict[char] for char in np.squeeze(np.append(data[:, 0], data[:, 1]))]
    print('Reversely mapped sample text', ''.join(data_reverse[0:200]))
    data_mask = new_mask  #data_mask[0:N_batch*size_batch].reshape((size_batch, N_batch, 1))
    data_targets = new_target  #data_targets[0:N_batch*size_batch].reshape((size_batch, N_batch, 1))

    assert(data.shape == data_targets.shape)
    assert(data.shape == data_mask.shape)
    assert(len(data.shape) == 3)
    assert(len(data_targets.shape) == 3)
    assert(len(data_mask.shape) == 3)

    return data, data_targets, data_mask


train_inputs, train_targets, train_mask = prepare_data(train, vocab_dict, size_batch)
valid_inputs, valid_targets, valid_mask = prepare_data(valid, vocab_dict, size_batch)
test_inputs, test_targets, test_mask = prepare_data(test, vocab_dict, size_batch)

# ----------------------------- Set up Network ------------------------------ #
n_classes = len(vocab_dict)

inp, out = bs.get_in_out_layers_for_classification(n_classes, n_classes, outlayer_name='out',
                                                   mask_name='mask')
inp >> bs.layers.Lstm(1000, name='lstm') >> out
# CLOCKWORK RNN:
# size_rnn = 1000
# timing = 1 * np.ones(100, dtype=int)
# for i in range(1, 10):
    # print(i)
    # timing = np.concatenate((timing, pow(2, i) * np.ones(100, dtype=int)), axis=0)
# print(timing) # Cw_Rnn
# inp >> bs.layers.ClockworkRnn(size_rnn, timing, name='cw_rnn') >> out
#inp >> bs.layers.FullyConnected(100, activation_function='linear') >> out
network = bs.Network.from_layer(out)
# network = bs.Network.from_hdf5('penn_corpus_best_lstm_49_new1.hdf5')

network.set_memory_handler(PyCudaHandler())
# network.initialize({"default": bs.Gaussian(0.1), "lstm": {'bf': 2, 'bi': 2, 'bo': 2}}, seed=42)
network.initialize({"default": bs.Gaussian(0.1)}, seed=42)
network.set_gradient_modifiers({"lstm": bs.ClipValues(low=-1., high=1)})

# ---------------------------- Set up Iterators ----------------------------- #

print(train_inputs.shape)
print(train_mask.shape)
print(train_targets.shape)


train_getter = bs.Minibatches(batch_size=100, verbose=True, mask=train_mask,
                              default=train_inputs, targets=train_targets,
                              seed=45252, shuffle=False)
valid_getter = bs.Minibatches(batch_size=100, verbose=True, mask=valid_mask,
                              default=valid_inputs, targets=valid_targets, shuffle=False)
test_getter = bs.Minibatches(batch_size=100, verbose=True, mask=test_mask,
                             default=test_inputs, targets=test_targets, shuffle=False)
vocab_dict_name = {'default': len(vocab_dict)}
train_getter = One_hot(train_getter, vocab_dict_name)
print(train_getter)
valid_getter = One_hot(valid_getter, vocab_dict_name)
test_getter = One_hot(test_getter, vocab_dict)

print(train_getter.iter.data['default'].shape)
# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.MomentumStep(learning_rate=0.01, momentum=0.99), double_buffering=False)
trainer.add_hook(bs.hooks.StopAfterEpoch(500))
trainer.add_hook(bs.hooks.MonitorAccuracy("valid_getter",
                                          output="out.output",
                                          mask_name="mask",
                                          name="validation",
                                          verbose=True))
trainer.add_hook(bs.hooks.SaveBestNetwork("validation.accuracy",
                                          filename='penn_corpus_best_lstm_49_new1.hdf5',
                                          name="best weights",
                                          criterion="max"))
trainer.add_hook(bs.hooks.MonitorLayerParameters('lstm', verbose=False))
trainer.add_hook(bs.hooks.MonitorLayerGradients('lstm', timescale='update', verbose=False))

# -------------------------------- Train ------------------------------------ #
trainer.train(network, train_getter, valid_getter=valid_getter)
print("\nBest validation accuracy: ", max(trainer.logs["validation accuracy"]))
