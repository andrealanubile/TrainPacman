import json
from channels.generic.websocket import AsyncWebsocketConsumer
import redis

class PacmanConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("pacman_group", self.channel_name)
        await self.accept()
        print('WebSocket connection accepted.')

        self.redis = redis.Redis(host='localhost', port=6379, db=0)

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("pacman_group", self.channel_name)

    async def receive(self, text_data):
        message = json.loads(text_data)
        action = message.get('action')

        if action:
            self.redis.lpush('rewards', action)

    async def state_update(self, event):
        pacman_loc = event['pacman_loc']
        pacman_direction = event['pacman_direction']
        ghost_loc = event['ghost_loc']
        ghost_direction = event['ghost_direction']
        pellets = event['pellets']
        await self.send(text_data=json.dumps({'pacman_loc': pacman_loc,
                                              'pacman_direction': pacman_direction,
                                              'ghost_loc': ghost_loc,
                                              'ghost_direction': ghost_direction,
                                              'pellets': pellets}))