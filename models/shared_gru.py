import collections

import torch
from torch import nn
import torch.nn.functional as F

import utils

import models.shared_base
from models.shared_rnn import EmbeddingDropout, LockedDropout, _get_dropped_weights

logger = utils.get_logger()


class GRU(models.shared_base.SharedModel):
    """Shared RNN model."""
    def __init__(self, args, corpus):
        models.shared_base.SharedModel.__init__(self)

    def forward(self, inputs, dag, hidden=None, is_train=True):
        pass

    def cell(self, x, h_prev, dag):
        """Computes a single pass through the discovered GRU cell."""
        pass

    def init_hidden(self, batch_size):
        pass

    def get_f(self, name):
        name = name.lower()
        if name == 'relu':
            f = F.relu
        elif name == 'tanh':
            f = F.tanh
        elif name == 'identity':
            f = lambda x: x
        else:
            f = F.sigmoid
        return f

    def get_num_cell_parameters(self, dag):
        pass

    def reset_parameters(self):
        pass
