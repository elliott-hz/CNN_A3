#!/bin/bash

# Dog Emotion Recognition Web Application - Startup Script
# Supports: iMac (Apple Silicon MPS) and MacBook Pro (Intel CPU)

echo "=========================================="
echo "🐕 Dog Emotion Recognition Web App"
echo "=========================================="
echo ""

# Detect Mac architecture
ARCH="$(uname -m)"
case "$ARCH" in
    arm64)
        echo "🖥️  Detected: Apple Silicon (M1/M2/M3) - Will use MPS GPU acceleration"
        CHIP_TYPE="apple_silicon"
        ;;
    x86_64)
        echo "🖥️  Detected: Intel Chip - Will use CPU inference"
        CHIP_TYPE="intel"
        ;;
    *)
        echo "⚠️  Unknown architecture: $ARCH (attempting to continue)"
        CHIP_TYPE="unknown"
        ;;
esac
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

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
    echo "   Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "   Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "   ℹ️  No virtual environment found, using system Python"
fi

# Install dependencies if needed
echo "   Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "   Installing dependencies..."
    pip3 install -r requirements.txt
elif ! python3 -c "import websockets" 2>/dev/null; then
    echo "   Installing WebSocket support..."
    pip3 install websockets>=12.0
fi

# Add Python user scripts to PATH (macOS specific)
PYTHON_USER_BIN="$HOME/Library/Python/3.9/bin"
if [ -d "$PYTHON_USER_BIN" ]; then
    export PATH="$PYTHON_USER_BIN:$PATH"
fi
# Also check for newer Python versions
for pyver in 3.10 3.11 3.12; do
    PYTHON_USER_BIN="$HOME/Library/Python/$pyver/bin"
    if [ -d "$PYTHON_USER_BIN" ]; then
        export PATH="$PYTHON_USER_BIN:$PATH"
    fi
done

# Find uvicorn command
UVICORN_CMD=""
if command -v uvicorn &> /dev/null; then
    UVICORN_CMD="uvicorn"
else
    # Check common macOS locations
    for path in \
        "$HOME/Library/Python/3.9/bin/uvicorn" \
        "$HOME/Library/Python/3.10/bin/uvicorn" \
        "$HOME/Library/Python/3.11/bin/uvicorn" \
        "$HOME/Library/Python/3.12/bin/uvicorn" \
        "/usr/local/bin/uvicorn" \
        "/opt/homebrew/bin/uvicorn"; do
        if [ -f "$path" ]; then
            UVICORN_CMD="$path"
            break
        fi
    done
    
    if [ -z "$UVICORN_CMD" ]; then
        echo "   ⚠️  Warning: uvicorn not found, using python3 -m uvicorn"
        UVICORN_CMD="python3 -m uvicorn"
    fi
fi

# Add Node.js to PATH if installed in home directory or common locations
NODEJS_FOUND=false
if [ -d "$HOME/nodejs/bin" ]; then
    export PATH="$HOME/nodejs/bin:$PATH"
    NODEJS_FOUND=true
elif command -v node &> /dev/null; then
    NODEJS_FOUND=true
elif [ -f "/usr/local/bin/node" ]; then
    export PATH="/usr/local/bin:$PATH"
    NODEJS_FOUND=true
elif [ -f "/opt/homebrew/bin/node" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
    NODEJS_FOUND=true
fi

# Check PyTorch device support based on chip type
echo "   Checking GPU acceleration..."
python3 -c "
import torch
print(f'   PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print('   ✅ GPU (CUDA) detected')
    device_type = 'GPU (CUDA)'
elif torch.backends.mps.is_available():
    print('   ✅ GPU (MPS - Apple Silicon) detected')
    device_type = 'GPU (MPS - Apple Silicon)'
else:
    print('   ℹ️  Running on CPU (no GPU available)')
    device_type = 'CPU'
    
# Verify device matches expected hardware
import platform
machine = platform.machine()
if machine == 'arm64':
    if device_type != 'GPU (MPS - Apple Silicon)':
        print('   ⚠️  Warning: Apple Silicon detected but MPS not available!')
        print('      Ensure PyTorch >= 2.0 is installed')
elif machine == 'x86_64':
    if device_type != 'CPU':
        print('   ℹ️  Intel Mac running on:', device_type)
"
echo ""

# Start the API server in the background
echo "   Starting FastAPI server..."
cd "$SCRIPT_DIR/api_service"
$UVICORN_CMD main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "   Backend PID: $BACKEND_PID"
echo "   Waiting for backend to start..."
sleep 5

# Check if backend started successfully
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend started successfully at http://localhost:8000"
    # Show device info from health endpoint
    DEVICE_INFO=$(curl -s http://localhost:8000/health | python3 -c "import sys, json; print(json.load(sys.stdin).get('device', 'Unknown'))" 2>/dev/null)
    echo "   Device: $DEVICE_INFO"
else
    echo "⚠️  Backend may not have started properly. Check the logs above."
fi

echo ""

# Display performance expectations based on chip type
if [ "$CHIP_TYPE" = "apple_silicon" ]; then
    echo "💡 Performance Expectations (Apple Silicon):"
    echo "   • Image inference: ~0.3-0.5 seconds per image"
    echo "   • Video processing: ~2-4 FPS (5 frames/sec extraction)"
    echo "   • Live streaming: ~5-10 FPS achievable"
    echo "   • GPU acceleration: MPS backend active ✅"
elif [ "$CHIP_TYPE" = "intel" ]; then
    echo "💡 Performance Expectations (Intel CPU):"
    echo "   • Image inference: ~1-2 seconds per image"
    echo "   • Video processing: ~1-2 FPS (5 frames/sec extraction)"
    echo "   • Live streaming: Not recommended (too slow)"
    echo "   • Note: Consider using cloud GPU for better performance"
fi
echo ""

# Start Frontend
echo "🎨 Starting Frontend (React + Vite)..."
cd "$SCRIPT_DIR/web_intf"

# Find npm executable
NPM_CMD=""
if command -v npm &> /dev/null; then
    NPM_CMD="npm"
elif [ -f "/usr/local/bin/npm" ]; then
    NPM_CMD="/usr/local/bin/npm"
elif [ -f "/opt/homebrew/bin/npm" ]; then
    NPM_CMD="/opt/homebrew/bin/npm"
elif [ -f "$HOME/nodejs/bin/npm" ]; then
    NPM_CMD="$HOME/nodejs/bin/npm"
else
    echo ""
    echo "⚠️  WARNING: Node.js and npm are not installed!"
    echo "   The frontend cannot start without Node.js."
    echo ""
    echo "   Please install Node.js from: https://nodejs.org/"
    echo "   Recommended: Download the LTS version (Long Term Support)"
    echo ""
    echo "   After installation, run this script again."
    echo ""
    echo "✅ Backend API is running at: http://localhost:8000"
    echo "   You can access the API documentation at: http://localhost:8000/docs"
    echo ""
    
    # Wait for backend process
    wait $BACKEND_PID
    exit 0
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "   Installing npm dependencies using $NPM_CMD..."
    $NPM_CMD install
fi

# Start the dev server
$NPM_CMD run dev &
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
echo "Live Stream:  ws://localhost:8000/ws/live-stream"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=========================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT
trap cleanup SIGTERM

# Wait for processes
wait
