from django.core.management.base import BaseCommand
from boxtest.tasks import update_x_coordinate

class Command(BaseCommand):
    help = 'Starts the update_x_coordinate task'

    def handle(self, *args, **kwargs):
        update_x_coordinate.delay()
        self.stdout.write(self.style.SUCCESS('Started update_x_coordinate task'))