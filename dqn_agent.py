import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from dqn_model import DQN

class DQNAgent:
    def __init__(self, state_space, action_space, learning_rate=0.001, gamma=0.99, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        
        self.memory = deque(maxlen=100000)
        self.model = DQN(state_space, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            q_update = reward
            if not done:
                q_update = (reward + self.gamma * torch.max(self.model(next_state)).item())
            q_values = self.model(state)
            q_values[0][action] = q_update
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, torch.FloatTensor(q_values.detach().numpy()))
            loss.backward()
            self.optimizer.step()
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)
