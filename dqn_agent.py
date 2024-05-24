import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from PERMemory import PERMemory

from dqn_model import DQN

class DQNAgent:
    def __init__(self, input_shape, num_actions, learning_rate=0.0001, gamma=0.99, exploration_max=1.0, exploration_min=0.1, exploration_decay=0.999):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.device = torch.device("cuda")

        self.memory = PERMemory(capacity=1000000, alpha=0.6)
        # self.memory = deque(maxlen=1000000)
        self.model_target = DQN(input_shape, num_actions).to(self.device)
        self.model_training = DQN(input_shape, num_actions).to(self.device)
        self.model_target.load_state_dict(self.model_training.state_dict())
        self.optimizer = optim.Adam(self.model_training.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.actions_mapping = {
            0: 'UP',
            1: 'DOWN',
            2: 'LEFT',
            3: 'RIGHT'
        }

    # def remember(self, state, action, reward, next_state, done):
    #     state = np.array(state, dtype=np.float32).squeeze()
    #     next_state = np.array(next_state, dtype=np.float32).squeeze()
    #     self.memory.append((state, action, reward, next_state, done))
    #     # print(len(self.memory))

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32).squeeze()
        next_state = np.array(next_state, dtype=np.float32).squeeze()
        transition = (state, action, reward, next_state, done)
        self.memory.add(error=1.0, sample=transition)  # Initial priority is max

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.choice(list(self.actions_mapping.values()))
        state = np.array(state, dtype=np.float32)
        state_t = torch.FloatTensor(state).to(self.device)
        if len(state_t.size()) != 4:
            state_t = state_t.unsqueeze(1)
        else:
            state_t = torch.permute(state_t,(1,0,2,3))

        q_values = self.model_target(state_t)
        action = torch.argmax(q_values).item()
        return self.actions_mapping[action]

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.model_target.load_state_dict(self.model_training.state_dict())
        print('Target Model Updated')

    # def experience_replay(self, batch_size):
    #     # self.optimizer = optim.Adam(self.model_training.parameters(), lr=self.learning_rate)

    #     if len(self.memory) < batch_size:
    #         return
    #     batch = random.sample(self.memory, batch_size)
    #     states, actions, rewards, next_states, dones = zip(*batch)

    #     # Reshape states to (batch_size, channels, height, width)
    #     states = torch.FloatTensor(np.array(states)).to(self.device)
    #     next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
    #     rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
    #     dones = torch.FloatTensor(np.array(dones)).to(self.device)

    #     if len(states.size()) != 4:
    #         states = states.unsqueeze(1)
    #         next_states = next_states.unsqueeze(1)

    #     q_values = self.model_training(states)
    #     next_q_values = self.model_target(next_states)

    #     target_q_values = torch.zeros_like(q_values)

    #     for i in range(batch_size):
    #         action_idx = list(self.actions_mapping.keys())[list(self.actions_mapping.values()).index(actions[i])]
    #         if dones[i]:
    #             target_q_values[i][action_idx] = rewards[i]
    #         else:
    #             target_q_values[i][action_idx] = rewards[i] + self.gamma * torch.max(next_q_values[i])

    #     self.optimizer.zero_grad()
    #     loss = self.criterion(q_values, target_q_values)
    #     loss.backward()
    #     self.optimizer.step()

    def experience_replay(self, batch_size):
        if len(self.memory.tree.data) < batch_size:
            return
        batch, idxs, is_weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Ensure that states and next_states are numpy arrays of the correct type and shape
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        # Ensure correct shapes: (batch_size, channels, height, width)
        states = states.reshape(batch_size, *self.input_shape)
        next_states = next_states.reshape(batch_size, *self.input_shape)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        q_values = self.model_training(states)
        next_q_values = self.model_training(next_states)
        target_q_values = q_values.clone()

        errors = []
        for i in range(batch_size):
            action_idx = list(self.actions_mapping.keys())[list(self.actions_mapping.values()).index(actions[i])]
            if dones[i]:
                target_q_values[i][action_idx] = rewards[i]
            else:
                target_q_values[i][action_idx] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()
            
            errors.append(abs(q_values[i][action_idx].item() - target_q_values[i][action_idx].item()))

        self.optimizer.zero_grad()
        loss = (is_weights * self.criterion(q_values, target_q_values)).mean()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        for idx, error in zip(idxs, errors):
            self.memory.update(idx, error)


    def save_model(self, path):
        """Save the model parameters to the specified path."""
        torch.save(self.model_training.state_dict(), path)

