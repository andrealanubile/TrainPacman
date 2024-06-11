from django.core.management.base import BaseCommand
from training_backend.tasks import run_train

class Command(BaseCommand):
    help = 'Starts the run_train task'

    def handle(self, *args, **kwargs):
        run_train.delay()
        self.stdout.write(self.style.SUCCESS('Started run_train task'))