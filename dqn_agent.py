import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from dqn_model import DQN

class DQNAgent:
    def __init__(self, input_shape, num_actions, learning_rate=0.0001, gamma=0.99, exploration_max=1.0, exploration_min=0.1, exploration_decay=0.995):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.device = torch.device("cuda")

        self.memory = deque(maxlen=100000)
        self.model_target = DQN(input_shape, num_actions).to(self.device)
        self.model_training = DQN(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model_training.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.actions_mapping = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(list(self.actions_mapping.values()))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model_target(state)
        action = torch.argmax(q_values).item()
        return self.actions_mapping[action]

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.model_target.load_state_dict(self.model_training.state_dict())
        print('Target Model Updated')


    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model_training(states)
        next_q_values = self.model_training(next_states)
        target_q_values = q_values.clone()

        for i in range(batch_size):
            action_idx = list(self.actions_mapping.keys())[list(self.actions_mapping.values()).index(actions[i])]
            if dones[i]:
                target_q_values[i][action_idx] = rewards[i]
            else:
                target_q_values[i][action_idx] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
            """Save the model parameters to the specified path."""
            torch.save(self.model_training.state_dict(), path)

