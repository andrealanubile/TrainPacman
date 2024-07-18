from training_backend.game_controller import GameController
from training_backend.dqn_model import DQN, ReplayMemory, Transition

import os
import math
import time
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
from hydra_zen import launch, store, builds, zen, multirun
from hydra.conf import HydraConf, JobConf, RunDir, SweepDir

global debug
debug = False


def optimize_model(memory, BATCH_SIZE, GAMMA, optimizer, policy_net, target_net, device):
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

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def select_action(state, exploration_rate, policy_net, n_actions, device):
    sample = random.random()
    if sample > exploration_rate:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(np.random.choice(np.arange(n_actions)), device=device, dtype=torch.long).view(1, 1)


def run_train(BATCH_SIZE, GAMMA, EPS_START, EPS_END, REPLAY_SIZE, TAU, LR, NUM_EPISODES, HORIZON, LEVEL):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    np.set_printoptions(threshold=sys.maxsize, linewidth=200)

    writer = SummaryWriter(log_dir='.')

    game = GameController(debug,LEVEL, reward_type='pretrain', render=True)
    game.startGame()

    state_dim = (4, len(game.rows_use), len(game.cols_use))  # assuming grid size (channels, height, width)
    n_actions = 4  # UP, DOWN, LEFT, RIGHT

    policy_net = DQN(state_dim, n_actions).to(device)
    target_net = DQN(state_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(REPLAY_SIZE)

    steps_done = 0

    episode_rewards = []
    episode_eps = []
    episode_scores = []
    episode_lengths = []

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    EPS_DECAY = NUM_EPISODES / np.log(EPS_START / EPS_END)

    for i_episode in tqdm(range(NUM_EPISODES)):
        # Initialize the environment and get its state
        if i_episode > 0:
            game.restartGame()
        state = game.get_state()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        exploration_rate = EPS_START * math.exp(-1. * i_episode / EPS_DECAY)
        force_episode_end = False
        for t in count():
            action = select_action(state, exploration_rate, policy_net, n_actions, device)
            reward, next_state, done = game.update(action.item())
            if force_episode_end:
                reward = -250
                done = True
            reward = torch.tensor([reward], device=device)
            episode_reward += reward
            episode_score = torch.tensor([game.score], device=device)
            # print(episode_reward)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
                
            memory.push(state, action, next_state, reward)
                

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, BATCH_SIZE, GAMMA, optimizer, policy_net, target_net, device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if HORIZON is not None:
                if t > HORIZON:
                    force_episode_end = True

            if done:
                # episode_rewards.append(episode_reward)
                # episode_eps.append(torch.tensor([exploration_rate], device=device))
                # episode_scores.append(episode_score)
                # episode_lengths.append(t)
                # plot_rewards()
                writer.add_scalar('Reward', episode_reward, i_episode)
                writer.add_scalar('Exploration rate', exploration_rate, i_episode)
                writer.add_scalar('Score', episode_score, i_episode)
                writer.add_scalar('Episode length', t, i_episode)
                break

        if i_episode % 500 == 0:
            torch.save(policy_net.state_dict(), f'dqn_checkpoint_iter_{i_episode}.pt') 

    torch.save(policy_net.state_dict(), 'dqn_model.pt')
    memory.save_memory('replay_memory.pkl')
    writer.close()


store(builds(run_train, 
             BATCH_SIZE=128, 
             GAMMA=0.99,
             EPS_START=0.9,
             EPS_END=0.05,
             REPLAY_SIZE=10000,
             TAU=0.005,
             LR=1e-4,
             NUM_EPISODES=3000,
             HORIZON=10000,
             LEVEL=0), name='run_train')

if __name__ == '__main__':
    store(HydraConf(job=JobConf(chdir=True),
                    run=RunDir(dir='hydra_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.name}'),
                    sweep=SweepDir(dir='hydra_runs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.name}', subdir='job_${hydra.job.num}_${hydra.job.override_dirname}')),
                    name="config", group="hydra")
    store.add_to_hydra_store(overwrite_ok=True)
    task_fn = zen(run_train)

    job = launch(store[None, 'run_train'],
                 task_fn,
                 overrides={
                    'BATCH_SIZE': 128,
                    'LR': 1e-4,
                    'TAU': 0.005,
                    'NUM_EPISODES': multirun([300, 500, 1000]),
                    'EPS_END': 0.01
                 },
                 job_name='DQN_tuning',
                 multirun=True,
                 version_base='1.2')
