from __future__ import absolute_import, unicode_literals
from celery import shared_task
import math
import time
import redis
import json
import numpy as np
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .game_controller import GameController

@shared_task
def update_state():
    r = redis.Redis(host='localhost', port=6379, db=0)
    channel_layer = get_channel_layer()

    game = GameController(debug=False)
    game.startGame()

    while True:
        action = np.random.choice(np.arange(4))
        _, state, _ = game.update(action)
        pacman_loc = json.dumps(game.pacman.getPos())
        pacman_direction = game.pacman.direction
        pellets = json.dumps(game.pellets.getList())
        r.set('pacman_loc', pacman_loc)
        r.set('pacman_direction', pacman_direction)
        r.set('pellets', pellets)
        async_to_sync(channel_layer.group_send)(
            'pacman_group',
            {'type': 'state_update',
             'pacman_loc': pacman_loc,
             'pacman_direction': pacman_direction,
             'pellets': pellets},
        )
        time.sleep(0.5)


    # while True:
    #     x_coordinate = 50 * (1 + math.sin(time.time()))
    #     r.set('x_coordinate', x_coordinate)
    #     async_to_sync(channel_layer.group_send)(
    #         "box_group",
    #         {"type": "box_update", "x_coordinate": x_coordinate},
    #     )
    #     time.sleep(0.1)