from __future__ import absolute_import, unicode_literals
import os
import sys
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'train_pacman.settings')

app = Celery('train_pacman')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()