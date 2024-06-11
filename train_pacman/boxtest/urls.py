from django.urls import path
from .views import InitialCoordinateView

urlpatterns = [
    path('api/initial-coordinate/', InitialCoordinateView.as_view(), name='initial-coordinate'),
]