from django.core.management.base import BaseCommand
from training_backend.tasks import optimize_model

class Command(BaseCommand):
    help = 'Starts the optimize_model task'

    def handle(self, *args, **kwargs):
        optimize_model.delay()
        self.stdout.write(self.style.SUCCESS('Started optimize_model task'))