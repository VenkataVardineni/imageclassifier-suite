#!/bin/bash

# Start script for Image Classifier Suite
# This script starts both the backend API and frontend UI

echo "🚀 Starting Image Classifier Suite..."

# Check if model exists
if [ ! -f "./checkpoints/best.pth" ]; then
    echo "⚠️  Warning: Model checkpoint not found at ./checkpoints/best.pth"
    echo "   Please train the model first or update MODEL_PATH in src/api/main.py"
    echo ""
fi

# Start backend API in background
echo "📡 Starting FastAPI backend..."
python src/api/main.py &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend
echo "🎨 Starting React frontend..."
cd frontend
npm install > /dev/null 2>&1
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ Services started!"
echo "   Backend API: http://localhost:8000"
echo "   Frontend UI: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

