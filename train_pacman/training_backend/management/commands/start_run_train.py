from django.core.management.base import BaseCommand
from training_backend.tasks import run_train, optimize_model

class Command(BaseCommand):
    help = 'Starts the run_train and optimize_model tasks'

    def handle(self, *args, **kwargs):
        run_train.delay()
        optimize_model.delay()
        self.stdout.write(self.style.SUCCESS('Started run_train and optimize_model tasks'))