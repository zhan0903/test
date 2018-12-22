import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import torch.nn as nn
import torch


def create_init_particles():
    # Xavier initialization
    node_in = 8*8*4
    node_out = 32
    # W_xavier= np.random.randn(node_in, node_out) / np.sqrt(node_in)

    w = torch.empty([32, 4, 8, 8])
    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')

    print(w .flatten()[-100:])
    # print(len(w .flatten()))
    print(w.mean())

    # # He initialization
    # stddev = np.sqrt(2 / node_in)
    #
    # W_he = np.random.randn(node_in, node_out) / np.sqrt(node_in / 2)
    # print(W_he.flatten()[-100:])
    # print(len(W_he.flatten()))
    #
    # # random initialization
    # W_random = np.random.randn(node_in, node_out)*0.01
    # print(W_random.flatten()[-100:])
    #
    # # he_uniform initializaoion
    # limit = np.sqrt(6/node_in)
    # W_uniform = np.random.uniform(low=-limit, high=limit, size=node_in*node_out)
    # print(W_uniform[-100:])


def init_weight(shape, type):
    # torch.manual_seed(123)
    w = torch.empty(shape)
    if type == "xavier_normal":
        nn.init.xavier_normal_(w)
    if type == "xavier_uniform":
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    if type == "kaiming_uniform":
        nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    if type == "kaiming_normal":
        nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    if type == "orthogonal":
        nn.init.orthogonal_(w)
    if type == "bias":
        nn.init.constant_(w, 0)

    return w.numpy().flatten()


# create_init_particles([32, 4, 8, 8],"kaiming_uniform")
# print(init_weight([32, 4, 8, 8],"orthogonal")[-100:])
create_init_particles()


