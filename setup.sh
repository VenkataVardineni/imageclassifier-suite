#!/bin/bash

# Image Classifier Suite - Setup Script
# This script automates the setup process for the application

set -e  # Exit on error

echo "🚀 Image Classifier Suite - Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python found: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js found: $NODE_VERSION${NC}"
else
    echo -e "${RED}❌ Node.js not found. Please install Node.js 16+${NC}"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✅ npm found: $NPM_VERSION${NC}"
else
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi

echo ""
echo "📦 Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install Python dependencies
echo "Installing Python dependencies..."
if pip install -r requirements.txt --quiet; then
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${RED}❌ Failed to install Python dependencies${NC}"
    exit 1
fi

echo ""
echo "📦 Setting up Frontend..."

# Install frontend dependencies
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    if npm install --silent; then
        echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
    else
        echo -e "${RED}❌ Failed to install frontend dependencies${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  Frontend dependencies already installed${NC}"
fi

cd ..

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Train the model (if you don't have a checkpoint):"
echo "   python src/training/train.py --epochs 100"
echo ""
echo "2. Start the backend API (Terminal 1):"
echo "   source venv/bin/activate"
echo "   python src/api/main.py"
echo ""
echo "3. Start the frontend (Terminal 2):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "4. Open your browser: http://localhost:3000"
echo ""
echo "📚 For detailed instructions, see SETUP.md"
echo ""

