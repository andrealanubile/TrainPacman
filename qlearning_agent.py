import numpy as np
import random
from collections import defaultdict
import pickle
import gzip

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.5, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.999):
        self.action_space = action_space
        self.action_mapping = {i: action for i, action in enumerate(action_space)}
        self.reverse_action_mapping = {action: i for i, action in enumerate(action_space)}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)  # Explore
        else:
            action_index = np.argmax(self.q_table[state])  # Exploit
            return self.action_mapping[action_index]

    def update_q_values(self, state, action, reward, next_state):
        action_index = self.reverse_action_mapping[action]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action_index]
        
        # Print Q-learning update step details
        # print(f"State: {state}")
        # print(f"Action: {action} (index: {action_index})")
        # print(f"Reward: {reward}")
        # print(f"Next State: {next_state}")
        # print(f"TD Target: {td_target}")
        # print(f"TD Error: {td_error}")
        # print(f"Old Q-value: {self.q_table[state][action_index]}")
        
        self.q_table[state][action_index] += self.lr * td_error

        # Print the updated Q-value
        # print(f"Updated Q-value: {self.q_table[state][action_index]}\n")

        self.epsilon *= self.epsilon_decay  # Decay exploration rate

    def save_policy(self, file_path):
        # print(f"Saving Q-table with {len(self.q_table)} entries.")
        with open(file_path, 'wb') as file:
            pickle.dump(dict(self.q_table), file)

    def load_policy(self, file_path):
        with open(file_path, 'rb') as file:
            self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)), pickle.load(file))
        # print(f"Loaded Q-table with {len(self.q_table)} entries.")