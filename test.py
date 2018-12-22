from tensorboardX import SummaryWriter
import tensorflow as tf
import csv
import torch.nn as nn
import torch
import numpy as np
import math

# test = ["123", "456"]
# writer = SummaryWriter(comment="-test")
#
# for item in test:
#     writer = SummaryWriter(comment="-test%s" % item)
#
#     x = 10
#     # writer = SummaryWriter(comment="-test%s" % item)
#
#     for index in range(5):
#         x = 2*x
#         writer.add_scalar("best_agent_5 %s" % item, x, index)
#         # count = count + 1


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


# for item in range(10):
#     with open('./results/employee_file%s.csv' % "test11", mode='a') as employee_file:
#         employee_writer = csv.writer(employee_file, delimiter=',')
#         employee_writer.writerow([item, item+1])

shape_out = 6
net = Net((4, 84, 84), shape_out)
for p in net.modules():
    torch.manual_seed(12345671212)
    if isinstance(p, nn.Conv2d):
        nn.init.kaiming_normal_(p.weight.data, nonlinearity='relu')
        p.bias.data.zero_()
        # scale_by.append(p.weight.data.numpy().flatten().copy())
        # scale_by.append(p.bias.data.numpy().flatten().copy())
    if isinstance(p, nn.Linear):
        nn.init.kaiming_uniform_(p.weight.data, nonlinearity='relu')
        p.bias.data.zero_()
        # bound = 1 / math.sqrt(512)
        # nn.init.uniform_(p.bias, -bound, bound)
        print("p.bias.mean", p.bias.mean())
        # scale_by.append(p.weight.data.numpy().flatten().copy())


p = 0.5
# obs = torch.empty([1, 4, 84, 84])
obs = torch.empty([1, 4, 2])

print(obs.shape)
nn.init.kaiming_uniform_(obs, nonlinearity='relu')
print(*obs.shape)
U1 = (np.random.rand(*obs.shape) < p) / p
print(U1)

exit(0)

torch.manual_seed(97556346346)
nn.init.kaiming_uniform_(obs, nonlinearity='relu')
print(obs[-10:])
act_prob = net(obs)
print("act_prob", act_prob)
print("act_prob.mean", act_prob.mean())
acts = act_prob.max(dim=1)[1]
print("acts", acts)  # default 2

exit(0)

for p in net.modules():
    if isinstance(p, nn.Conv2d):
        nn.init.xavier_uniform_(p.weight.data, gain=nn.init.calculate_gain('relu'))
        p.bias.data.zero_()
        scale_by.append(p.weight.data.numpy().flatten().copy())
        scale_by.append(p.bias.data.numpy().flatten().copy())
    if isinstance(p, nn.Linear):
        nn.init.xavier_uniform_(p.weight.data, gain=nn.init.calculate_gain('relu'))
        p.bias.data.zero_()
        scale_by.append(p.weight.data.numpy().flatten().copy())
        scale_by.append(p.bias.data.numpy().flatten().copy())