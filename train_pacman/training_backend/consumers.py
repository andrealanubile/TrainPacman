import json
from channels.generic.websocket import AsyncWebsocketConsumer
import redis

class PacmanConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("pacman_group", self.channel_name)
        await self.accept()
        print('WebSocket connection accepted.')

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("pacman_group", self.channel_name)

    async def receive(self, text_data):
        print(f'Received message: {text_data}')  # Add print statement
        pass

    async def state_update(self, event):
        pacman_loc = event['pacman_loc']
        pacman_direction = event['pacman_direction']
        pellets = event['pellets']
        await self.send(text_data=json.dumps({'pacman_loc': pacman_loc,
                                              'pacman_direction': pacman_direction,
                                              'pellets': pellets}))