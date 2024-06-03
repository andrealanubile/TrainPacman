from rest_framework.views import APIView
from rest_framework.response import Response
import redis

class InitialStateView(APIView):
    def get(self, request):
        r = redis.Redis(host='localhost', port=6379, db=0)

        pacman_loc = r.get('pacman_loc')
        if pacman_loc is None:
            pacman_loc = (0, 0)
        return Response({'pacman_loc': pacman_loc})
