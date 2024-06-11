development:

npm run build
npm start
redis-server
daphne -p 8000 train_pacman.asgi:application
celery -A train_pacman worker --loglevel=info
python manage.py start_task

production:

npm run production
redis-server
daphne -b 0.0.0.0 -p 8000 train_pacman.asgi:application
celery -A train_pacman worker --loglevel=info
python manage.py start_task