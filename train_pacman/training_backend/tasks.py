from __future__ import absolute_import, unicode_literals
import math
import time
import redis
import json
import os
import io
import math
import pickle
from itertools import count
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .game_controller import GameController
from .dqn_model import DQN, ReplayMemory, ReplayMemoryRedis, Transition



@shared_task
def run_train():
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

    def plot_rewards(show_result=False):
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        scores_t = torch.tensor(episode_scores, dtype=torch.float)
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

        # Convert episode_eps to CPU and then to numpy for plotting
        ax2.set_ylabel('Score')
        ax2.plot(scores_t.numpy())
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax2.plot(means.numpy())

        plt.savefig(os.path.join('training_backend', 'results', 'rewards.png'))

    def convert_to_list(mixed_list):
        converted_list = []
        for item in mixed_list:
            if isinstance(item, torch.Tensor):
                converted_list.extend(item.cpu().numpy().tolist())
            elif isinstance(item, (list, tuple)):
                converted_list.extend(convert_to_list(item))  # Recursively handle nested lists
            else:
                converted_list.append(item)
        return converted_list

    def save_results():
        # Convert all tensors to lists
        episode_rewards_list = convert_to_list(episode_rewards)
        episode_scores_list = convert_to_list(episode_scores)

        # Create a DataFrame
        data = {
            'Episode Rewards': episode_rewards_list,
            'Episode Scores': episode_scores_list
        }
        df = pd.DataFrame(data)

        # Define the directory and filename
        results_dir = 'training_backend/results'
        csv_filename = f'Results_data_{LEVEL}.csv'
        
        # Construct the full file path
        file_path = os.path.join(results_dir, csv_filename)

        # Save DataFrame to CSV in the specified directory
        df.to_csv(file_path, index=False)

        print(f"Data saved to {file_path}")


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS = 0.05
    REPLAY_SIZE = 10000
    TAU = 0.005
    LR = 1e-4
    HORIZON = None
    LEVEL = 0
    DEBUG = False

    r = redis.Redis(host='localhost', port=6379, db=0)
    channel_layer = get_channel_layer()

    game = GameController(DEBUG, LEVEL, reward_type='hf')
    game.startGame()

    state_dim = (4, len(game.rows_use), len(game.cols_use))  # assuming grid size (channels, height, width)
    n_actions = 4  # UP, DOWN, LEFT, RIGHT

    r.delete('policy_net')
    r.delete('target_net')
    
    r.set('state_dim', pickle.dumps(state_dim))
    r.set('n_actions', n_actions)
    r.set('batch_size', BATCH_SIZE)
    r.set('gamma', GAMMA)
    r.set('lr', LR)
    r.set('tau', TAU)

    policy_net = DQN(state_dim, n_actions)
    prev_state_dict = policy_net.state_dict()

    memory = ReplayMemoryRedis(r, REPLAY_SIZE)
    memory.clear()

    num_episodes = 0

    episode_rewards = []
    episode_scores = []

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    time_interval = 0.6
    start_time = time.time()

    while True:
        if not r.exists('policy_net'):
            time.sleep(0.1)
            continue
        # Initialize the environment and get its state
        game.restartGame()
        state = game.get_state()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        episode_score = 0

        for t in count():
            policy_net_bytestr = r.get('policy_net')
            buffer_policy_net = io.BytesIO(policy_net_bytestr)
            buffer_policy_net.seek(0)
            policy_net.load_state_dict(torch.load(buffer_policy_net))


            action = select_action(state, EPS)
            reward, next_state, done = game.update(action.item())

            pacman_loc = json.dumps(game.pacman.getPos())
            pacman_direction = game.pacman.direction
            ghost_loc = json.dumps(game.ghosts.ghosts[0].getPos())
            ghost_direction = game.ghosts.ghosts[0].direction
            pellets = json.dumps(game.pellets.getList())
            r.set('pacman_loc', pacman_loc)
            r.set('pacman_direction', pacman_direction)
            r.set('ghost_loc', ghost_loc)
            r.set('ghost_direction', ghost_direction)
            r.set('pellets', pellets)
            r.set('num_lives', game.lives)
            r.set('score', game.score)
            async_to_sync(channel_layer.group_send)(
                'pacman_group',
                {'type': 'state_update',
                'pacman_loc': pacman_loc,
                'pacman_direction': pacman_direction,
                'ghost_loc': ghost_loc,
                'ghost_direction': ghost_direction,
                'pellets': pellets,
                'num_lives': game.lives,
                'score': game.score},
            )

            elapsed_time = time.time() - start_time
            if elapsed_time > time_interval:
                print(f'Warning: thread too slow: took {elapsed_time} seconds')

            time.sleep(max(0, time_interval - elapsed_time))

            start_time = time.time()

            received_input = False

            while r.llen('rewards') > 0:
                received_input = True
                hf_reward = r.rpop('rewards').decode('utf-8')
                if hf_reward == 'reward_plus1':
                    reward += 10
                elif hf_reward == 'reward_plus10':
                    reward += 100
                elif hf_reward == 'reward_minus1':
                    reward -= 10
                elif hf_reward == 'reward_minus10':
                    reward -= 100

            reward = torch.tensor([reward], device=device).clamp(-1000, 1000)

            episode_reward += reward
            episode_score += game.score

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

                
            if (reward != 0) or received_input:
                # Store the transition in memory
                memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            # optimize_model()

            # Move to the next state
            state = next_state

            if HORIZON is not None:
                if t > HORIZON:
                    episode_rewards.append(episode_reward)
                    episode_scores.append(episode_score)
                    break

            if done:
                episode_rewards.append(episode_reward)
                episode_scores.append(episode_score)
                break
        
        num_episodes += 1

        if num_episodes % 10 == 0:
            torch.save(policy_net.state_dict(), os.path.join('training_backend', 'models', 'model_checkpoint.pt'))
            save_results()
            plot_rewards()


@shared_task
def optimize_model():
    r = redis.Redis(host='localhost', port=6379, db=0)
    memory = ReplayMemoryRedis(r)

    while True:
        if r.exists('state_dim') and r.exists('n_actions'):
            state_dim = pickle.loads(r.get('state_dim'))
            n_actions = int(r.get('n_actions'))
            break
    
    pretrain_checkpoint = 'dqn_checkpoint_iter_3000.pt'

    policy_net = DQN(state_dim, n_actions)
    target_net = DQN(state_dim, n_actions)
    policy_net.load_state_dict(torch.load(os.path.join('training_backend', 'models', pretrain_checkpoint)))
    target_net.load_state_dict(policy_net.state_dict())

    buffer_policy_net = io.BytesIO()
    torch.save(policy_net.state_dict(), buffer_policy_net)
    buffer_policy_net.seek(0)
    policy_net_bytestr = buffer_policy_net.read()
    r.set('policy_net', policy_net_bytestr)

    BATCH_SIZE = int(r.get('batch_size'))
    GAMMA = float(r.get('gamma'))
    LR = float(r.get('lr'))
    TAU = float(r.get('tau'))

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    while True:
        if len(memory) < BATCH_SIZE:
            continue
        # if not (r.exists('policy_net_before') and r.exists('target_net_before')):
        #     continue
        
        # policy_net_bytestr = r.get('policy_net_before')
        # target_net_bytestr = r.get('target_net_before')
        # buffer_policy_net = io.BytesIO(policy_net_bytestr)
        # buffer_target_net = io.BytesIO(target_net_bytestr)
        # buffer_policy_net.seek(0)
        # buffer_target_net.seek(0)
        # policy_net.load_state_dict(torch.load(buffer_policy_net))
        # target_net.load_state_dict(torch.load(buffer_target_net))

        # r.delete('policy_net_before')
        # r.delete('target_net_before')


        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
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
        next_state_values = torch.zeros(BATCH_SIZE)
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

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

        buffer_policy_net = io.BytesIO()
        torch.save(policy_net_state_dict, buffer_policy_net)
        buffer_policy_net.seek(0)
        policy_net_bytestr = buffer_policy_net.read()
        r.set('policy_net', policy_net_bytestr)