from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # path('ws/box/', consumers.SineWaveConsumer.as_asgi()),
    re_path(r'ws/box/$', consumers.BoxConsumer.as_asgi()),
]