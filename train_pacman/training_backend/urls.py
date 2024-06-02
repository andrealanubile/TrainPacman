from django.urls import path
from .views import InitialStateView

urlpatterns = [
    path('api/initial-state/', InitialStateView.as_view(), name='initial-state'),
]