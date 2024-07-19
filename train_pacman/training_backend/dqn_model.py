import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN2(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN2, self).__init__()
        self.pad = nn.ZeroPad2d((0, 0, 1, 0))
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(num_actions)

    def forward(self, x):
        x = self.pad(x)
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

    def save_memory(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def __len__(self):
        return len(self.memory)


class ReplayMemoryRedis(object):
    def __init__(self, r, capacity, hf_capacity, hf_sample_max=8):
        self.capacity = capacity
        self.hf_capacity = hf_capacity
        self.r = r
        self.key = 'replay_memory'
        self.hf_key = 'replay_memory_hf'
        self.count_key = 'hf_sample_count'
        self.hf_sample_max = hf_sample_max

    def push(self, *args, hf=False):
        transition = Transition(*args)
        serialized_transition = pickle.dumps(transition)
        if hf:
            key = self.hf_key
            capacity = self.hf_capacity
        else:
            key = self.key
            capacity = self.capacity

        self.r.lpush(key, serialized_transition)
        # Ensure the memory does not exceed capacity
        if self.r.llen(key) > capacity:
            s = self.r.rpop(key)
            if hf:
                self.r.hdel(self.count_key, s)


    def sample(self, batch_size, batch_size_hf):
        length_hf = self.r.llen(self.hf_key)
        indices_hf = random.sample(range(length_hf), min(length_hf, batch_size_hf))
        serialized_samples_hf = [self.r.lindex(self.hf_key, i) for i in indices_hf]
        hf_samples = [pickle.loads(sample) for sample in serialized_samples_hf]

        length = self.r.llen(self.key)
        indices = random.sample(range(length), batch_size - len(hf_samples))
        serialized_samples = [self.r.lindex(self.key, i) for i in indices]
        normal_samples = [pickle.loads(sample) for sample in serialized_samples]

        # increase count of human feedback transitions sampled
        for s in serialized_samples_hf:
            self.r.hincrby(self.count_key, s, 1)
            count = self.r.hget(self.count_key, s)
            count = int(count) if count else 0
            if count > self.hf_sample_max:
                # delete from human feedback buffer
                self.r.lrem(self.hf_key, 1, s)
                self.r.hdel(self.count_key, s)
                # add to normal buffer
                self.push(*pickle.loads(s))

        return normal_samples + hf_samples

    def load_memory(self, filename, device):
        with open(filename, 'rb') as f:
            memory = pickle.load(f)

        self.clear()

        for transition in memory:
            self.r.rpush(self.key, pickle.dumps(Transition(*[item.to(device) if item is not None else None for item in transition])))


    def clear(self):
        self.r.delete(self.key)
        self.r.delete(self.hf_key)
        self.r.delete(self.count_key)
    
    def __len__(self):
        return self.r.llen(self.key)