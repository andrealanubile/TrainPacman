from __future__ import absolute_import, unicode_literals
from celery import shared_task
import math
import time
import redis
import json
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from game_controller import GameController

@shared_task
def update_state():
    r = redis.Redis(host='localhost', port=6379, db=0)
    channel_layer = get_channel_layer()

    game = GameController(debug=False)
    game.startGame()

    pacman_loc = json.dumps(game.pacman.position.asTuple())

    r.set('pacman_loc', )

    # while True:
    #     x_coordinate = 50 * (1 + math.sin(time.time()))
    #     r.set('x_coordinate', x_coordinate)
    #     async_to_sync(channel_layer.group_send)(
    #         "box_group",
    #         {"type": "box_update", "x_coordinate": x_coordinate},
    #     )
    #     time.sleep(0.1)