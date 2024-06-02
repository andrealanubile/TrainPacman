from rest_framework.views import APIView
from rest_framework.response import Response
import redis

class InitialStateView(APIView):
    def get(self, request):
        r = redis.Redis(host='localhost', port=6379, db=0)

        # x_coordinate = r.get('x_coordinate')
        # if x_coordinate is None:
        #     x_coordinate = 0
        # return Response({'x_coordinate': float(x_coordinate)})
