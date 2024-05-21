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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory = deque(maxlen=100000)
        self.model = DQN(state_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.actions_mapping = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }
        self.inverse_actions_mapping = {v: k for k, v in self.actions_mapping.items()}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(list(self.actions_mapping.values()))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return self.actions_mapping[torch.argmax(q_values).item()]

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        target_q_values = q_values.clone().detach()

        for i in range(batch_size):
            q_update = rewards[i]
            if not dones[i]:
                q_update = (rewards[i] + self.gamma * torch.max(next_q_values[i]).item())
            target_q_values[i][self.inverse_actions_mapping[actions[i]]] = q_update
        
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
