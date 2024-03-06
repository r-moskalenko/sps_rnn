"""A module with controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils
from utils import Node


def _construct_dags(prev_nodes, activations, func_names, num_blocks):
    dags = []
    for nodes, func_ids in zip(prev_nodes, activations):
        dag = collections.defaultdict(list)

        # add first node
        dag[-1] = [Node(0, func_names[func_ids[0]])]
        dag[-2] = [Node(0, func_names[func_ids[0]])]

        # add following nodes
        for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
            dag[utils.to_item(idx)].append(Node(jdx + 1, func_names[func_id]))

        leaf_nodes = set(range(num_blocks)) - dag.keys()

        # merge with avg
        for idx in leaf_nodes:
            dag[idx] = [Node(num_blocks, 'avg')]

        last_node = Node(num_blocks + 1, 'h[t]')
        dag[num_blocks] = [last_node]
        dags.append(dag)

    return dags


class Controller(torch.nn.Module):
    def __init__(self, args, network_type, shared_rnn_activations, num_blocks, shared_cnn_types, controller_hid, is_cuda,
                 softmax_temperature, mode):
        torch.nn.Module.__init__(self)
        self.args = args
        self.num_blocks = num_blocks
        self.controller_hid = controller_hid
        self.is_cuda = is_cuda
        self.softmax_temperature = softmax_temperature
        self.mode = mode

        if network_type == 'rnn':
            # NOTE(brendan): `num_tokens` here is just the activation function
            # for every even step,
            self.num_tokens = [len(shared_rnn_activations)]
            for idx in range(num_blocks):
                self.num_tokens += [idx + 1,
                                    len(shared_rnn_activations)]
            self.func_names = shared_rnn_activations
        elif network_type == 'cnn':
            self.num_tokens = [len(shared_cnn_types), num_blocks]
            self.func_names = shared_cnn_types

        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens, controller_hid)
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, controller_hid),
                is_cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.softmax_temperature

        # exploration
        if self.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []
        for block_idx in range(2*(self.num_blocks - 1) + 1):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 2
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:mode]),
                requires_grad=False)

            if mode == 0:
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        dags = _construct_dags(prev_nodes,
                               activations,
                               self.func_names,
                               self.num_blocks)

        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.is_cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.is_cuda, requires_grad=False))
