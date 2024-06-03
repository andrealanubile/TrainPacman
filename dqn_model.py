# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class DQN(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

#         conv_out_size = self._get_conv_out(input_shape)
#         self.fc1 = nn.Linear(conv_out_size, 512)
#         self.fc2 = nn.Linear(512, num_actions)

#     def _get_conv_out(self, shape):
#         o = torch.zeros(1, *shape)
#         o = self.conv1(o)
#         o = self.conv2(o)
#         o = self.conv3(o)
#         return int(np.prod(o.size()))

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
