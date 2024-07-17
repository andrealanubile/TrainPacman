from training_backend.game_controller import GameController
from training_backend.dqn_model import DQN
import torch
import numpy as np
import sys
import os
import time
from tqdm import tqdm
from itertools import count

debug = False
LEVEL = 0
NUM_EPISODES = 100
HORIZON = 500

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)

np.set_printoptions(threshold=sys.maxsize, linewidth=200)


game = GameController(debug,LEVEL, reward_type='pretrain', render=True)
game.startGame()

state_dim = (4, len(game.rows_use), len(game.cols_use))  # assuming grid size (channels, height, width)
n_actions = 4  # UP, DOWN, LEFT, RIGHT

policy_net = DQN(state_dim, n_actions).to(device)
policy_net.load_state_dict(torch.load('dqn_model.pt', map_location=device))
policy_net.eval()

steps_done = 0

def select_action(state, policy_net):
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1).indices.view(1, 1)

for i_episode in tqdm(range(NUM_EPISODES)):
    # Initialize the environment and get its state
    if i_episode > 0:
        game.restartGame()
    state = game.get_state()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    for t in count():
        time.sleep(0.1)
        print(policy_net(state))
        action = select_action(state, policy_net)
        print(action)
        reward, next_state, done = game.update(action.item())
        if done:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        state = next_state

        if HORIZON is not None:
            if t > HORIZON:
                break

        if done:
            break