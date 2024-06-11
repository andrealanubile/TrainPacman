import json
from channels.generic.websocket import AsyncWebsocketConsumer
import redis

class BoxConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("box_group", self.channel_name)
        await self.accept()
        print('WebSocket connection accepted.')

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("box_group", self.channel_name)

    async def receive(self, text_data):
        print(f'Received message: {text_data}')  # Add print statement
        pass

    async def box_update(self, event):
        x_coordinate = event['x_coordinate']
        print(f'Sending x_coordinate: {x_coordinate}')  # Add print statement
        await self.send(text_data=json.dumps({'x_coordinate': x_coordinate}))