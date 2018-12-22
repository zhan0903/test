import torch.nn as nn
import torch
import numpy as np
import math
import pytest



class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


def worker(params_iterator):
    while True:
        params_iterator = params_iterator+1
    return params_iterator + 1


def test_worker_in_mpi():
    params_iterator = [1, 2, 3, 4]
    mp = torch.multiprocessing.get_context('forkserver')
    pool = mp.Pool(processes=1) # ,maxtasksperchild=1
    result = pool.map(worker, params_iterator)
    pool.close()
    pool.join()

    assert len(result) == 4
    assert sum(result) == 14

