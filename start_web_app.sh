#!/bin/bash

# Dog Emotion Recognition Web Application - Startup Script
# This script starts both the backend API and frontend dev server

echo "=========================================="
echo "🐕 Dog Emotion Recognition Web App"
echo "=========================================="
echo ""

# Get the directory where this script is located (this is the project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"  # Script is already in project root

# Check if models exist
echo "📦 Checking model files..."
if [ ! -f "$PROJECT_ROOT/best_models/detection_YOLOv8_baseline.pt" ]; then
    echo "❌ Error: Detection model not found at $PROJECT_ROOT/best_models/detection_YOLOv8_baseline.pt"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/best_models/emotion_ResNet50_baseline.pth" ]; then
    echo "❌ Error: Classification model not found at $PROJECT_ROOT/best_models/emotion_ResNet50_baseline.pth"
    exit 1
fi

echo "✅ Model files found"
echo ""

# Start Backend API
echo "🚀 Starting Backend API (FastAPI)..."
cd "$SCRIPT_DIR/api_service"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "   Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "   Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the API server in the background
python main.py &
BACKEND_PID=$!

echo "   Backend PID: $BACKEND_PID"
echo "   Waiting for backend to start..."
sleep 3

# Check if backend started successfully
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend started successfully at http://localhost:8000"
else
    echo "⚠️  Backend may not have started properly. Check the logs above."
fi

echo ""

# Start Frontend
echo "🎨 Starting Frontend (React + Vite)..."
cd "$SCRIPT_DIR/web_intf"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "   Installing npm dependencies..."
    npm install
fi

# Start the dev server
npm run dev &
FRONTEND_PID=$!

echo "   Frontend PID: $FRONTEND_PID"
echo ""

echo "=========================================="
echo "✅ Application Started!"
echo "=========================================="
echo ""
echo "Backend API:  http://localhost:8000"
echo "Frontend App: http://localhost:5173"
echo "API Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Wait for processes
wait
