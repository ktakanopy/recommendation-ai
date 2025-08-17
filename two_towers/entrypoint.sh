#!/bin/sh

# Wait for database to be ready
echo "Waiting for database to be ready..."
sleep 5

# Run database migrations
echo "Running database migrations..."
poetry run alembic upgrade head

# Start the application
echo "Starting the application..."
poetry run uvicorn --host 0.0.0.0 --port 8000 two_towers.app:app
