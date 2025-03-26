#!/bin/bash

# Kill any existing processes on ports 8000 and 3000
echo "Stopping any existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

# Start the backend server in the background
echo "Starting the backend server on port 8000..."
cd "$(dirname "$0")"
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 3

# Start the frontend
echo "Starting the frontend on port 3000..."
cd frontend
npm start

# If frontend is killed, also kill the backend
trap "kill $BACKEND_PID" EXIT

echo "Application started!"
echo "Backend PID: $BACKEND_PID"
echo "Access the application at http://localhost:3000"