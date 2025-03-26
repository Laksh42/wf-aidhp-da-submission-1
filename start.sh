#!/bin/bash

# Function to show help message
show_help() {
  echo "Usage: ./start.sh [options]"
  echo "Options:"
  echo "  -h, --help     Show this help message"
  echo "  -l, --local    Use local MongoDB (default: false)"
  echo "  -r, --remote   Use remote MongoDB (default: true)"
}

# Default settings
USE_LOCAL_DB=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -l|--local)
      USE_LOCAL_DB=true
      shift
      ;;
    -r|--remote)
      USE_LOCAL_DB=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Update .env file with database toggle setting
sed -i.bak "s/^USE_LOCAL_DB=.*/USE_LOCAL_DB=$USE_LOCAL_DB/" .env

# Kill any existing processes on ports 8000 and 3000
echo "Stopping any existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
fi

# Check if MongoDB is running locally
if $USE_LOCAL_DB; then
  echo "Using local MongoDB connection"
  if ! pgrep -x mongod > /dev/null; then
    echo "Warning: MongoDB is not running locally. Please start MongoDB service if needed."
  fi
else
  echo "Using remote MongoDB connection to MongoDB Atlas"
fi

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