from __future__ import absolute_import, unicode_literals
from celery import shared_task
import math
import time
import redis
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

@shared_task
def update_x_coordinate():
    r = redis.Redis(host='localhost', port=6379, db=0)
    channel_layer = get_channel_layer()
    while True:
        x_coordinate = 50 * (1 + math.sin(time.time()))
        r.set('x_coordinate', x_coordinate)
        async_to_sync(channel_layer.group_send)(
            "box_group",
            {"type": "box_update", "x_coordinate": x_coordinate},
        )
        time.sleep(0.1)