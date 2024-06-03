from training_backend.game_controller import GameController
from training_backend.dqn_model import DQN, ReplayMemory, Transition

import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm

global debug
debug = False


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



def plot_rewards(show_result=False):
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        ax1.set_title('Result')
    else:
        ax1.clear()
        ax1.set_title('Training...')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax1.plot(means.numpy())

    ax2.set_ylabel('Exploration rate')
    ax2.plot(episode_eps)

    plt.pause(0.001)  # pause a bit so that plots are updated


def select_action(state, exploration_rate):
    sample = random.random()
    if sample > exploration_rate:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(np.random.choice(np.arange(n_actions)), device=device, dtype=torch.long).view(1, 1)


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    np.set_printoptions(threshold=sys.maxsize, linewidth=200)

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 800
    REPLAY_SIZE = 10000
    TAU = 0.005
    LR = 1e-4
    NUM_EPISODES = 3000
    HORIZON = 200

    game = GameController(debug)
    game.startGame()

    state_dim = (3, len(game.rows_use), len(game.cols_use))  # assuming grid size (channels, height, width)
    n_actions = 4  # UP, DOWN, LEFT, RIGHT

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_SIZE)

    steps_done = 0

    episode_rewards = []
    episode_eps = []

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for i_episode in tqdm(range(NUM_EPISODES)):
        # Initialize the environment and get its state
        if i_episode > 0:
            game.restartGame()
        state = game.get_state()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        exploration_rate = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * i_episode / EPS_DECAY)
        for t in count():
            action = select_action(state, exploration_rate)
            reward, next_state, done = game.update(action.item())
            reward = torch.tensor([reward], device=device)
            episode_reward += reward

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if t is not None:
                if t > HORIZON:
                    episode_rewards.append(episode_reward)
                    episode_eps.append(exploration_rate)
                    plot_rewards()
                    break

            if done:
                episode_rewards.append(episode_reward)
                episode_eps.append(exploration_rate)
                plot_rewards()
                break

    torch.save(policy_net.state_dict(), 'dqn_model.pth')

    print('Complete')
    plot_rewards(show_result=True)
    plt.ioff()
    plt.show()
