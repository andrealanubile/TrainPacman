from django.core.management.base import BaseCommand
from training_backend.tasks import update_state

class Command(BaseCommand):
    help = 'Starts the update_state task'

    def handle(self, *args, **kwargs):
        update_state.delay()
        self.stdout.write(self.style.SUCCESS('Started update_state task'))