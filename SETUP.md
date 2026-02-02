# Setup Guide - Image Classifier Suite

Complete step-by-step guide to set up and run the Image Classifier Suite application.

## Prerequisites

Before starting, ensure you have the following installed:

- **Python 3.10 or higher**
- **Node.js 16+ and npm** (for the React frontend)
- **Git** (for cloning the repository)

### Check Your Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Check Node.js version
node --version   # Should be 16+

# Check npm version
npm --version
```

## Step 1: Clone the Repository

```bash
git clone https://github.com/VenkataVardineni/imageclassifier-suite.git
cd imageclassifier-suite
```

## Step 2: Python Environment Setup

### Option A: Using Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Global Installation

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```

## Step 4: Train the Model (Optional)

If you want to train your own model:

```bash
# Train with default settings (100 epochs)
python src/training/train.py

# Or train with custom settings
python src/training/train.py \
  --epochs 100 \
  --batch-size 128 \
  --scheduler cosine_warmup \
  --warmup-epochs 5
```

**Note**: Training will download CIFAR-10 dataset automatically (~170MB) on first run.

**Training Time**: 
- ~30 epochs: ~2-3 hours on Apple M2
- ~100 epochs: ~6-8 hours on Apple M2

The model checkpoints will be saved in `./checkpoints/`:
- `best.pth`: Best model based on validation accuracy
- `latest.pth`: Latest checkpoint

## Step 5: Verify Model Checkpoint

Ensure you have a trained model checkpoint:

```bash
# Check if checkpoint exists
ls -lh checkpoints/best.pth

# If it doesn't exist, you need to train first (see Step 4)
```

## Step 6: Start the Application

### Terminal 1: Start Backend API

```bash
# Make sure you're in the project root directory
cd /path/to/imageclassifier-suite

# Activate virtual environment if using one
source venv/bin/activate  # macOS/Linux
# or
# venv\Scripts\activate  # Windows

# Start the API server
python src/api/main.py
```

You should see:
```
Using device: mps  # or cuda/cpu
Model loaded from ./checkpoints/best.pth
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start Frontend

```bash
# Navigate to frontend directory
cd frontend

# Start React development server
npm start
```

The browser should automatically open to http://localhost:3000

If not, manually open: http://localhost:3000

## Step 7: Test the Application

1. **Open the web interface**: http://localhost:3000
2. **Upload an image**: 
   - Drag and drop an image onto the upload area, or
   - Click to select an image file
3. **View predictions**: See the classification results with confidence scores

### Test with API directly

```bash
# Health check
curl http://localhost:8000/health

# Test prediction (replace image.jpg with your image)
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"
```

## Quick Setup Script

For convenience, use the provided setup script:

```bash
# Make script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

This script will:
1. Check prerequisites
2. Create virtual environment (if needed)
3. Install Python dependencies
4. Install frontend dependencies
5. Provide instructions for starting the application

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3)

- PyTorch will automatically use MPS (Metal Performance Shaders)
- No additional configuration needed
- Training and inference are GPU-accelerated

### NVIDIA GPU (CUDA)

- Ensure CUDA-compatible PyTorch is installed
- The model will automatically use CUDA if available
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### CPU Only

- Works on any system
- Slower training and inference
- No additional setup required

## Directory Structure After Setup

```
imageclassifier-suite/
├── venv/                    # Virtual environment (if created)
├── node_modules/            # Frontend dependencies (in frontend/)
├── data/                    # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/             # Model checkpoints
│   ├── best.pth            # Best model
│   └── latest.pth          # Latest checkpoint
├── outputs/                 # Evaluation outputs
│   └── confusion_matrix.png
└── ... (other files)
```

## Troubleshooting

### Port Already in Use

If port 8000 or 3000 is already in use:

```bash
# Kill processes on ports
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### Model Not Found

If you see "Model not loaded" error:

1. Check if checkpoint exists: `ls checkpoints/best.pth`
2. Train the model if it doesn't exist (Step 4)
3. Or set MODEL_PATH environment variable:
   ```bash
   export MODEL_PATH=/path/to/your/model.pth
   python src/api/main.py
   ```

### Frontend Can't Connect to Backend

1. Ensure backend is running on port 8000
2. Check CORS settings in `src/api/main.py`
3. Verify proxy setting in `frontend/package.json`

### Dependencies Installation Issues

**Python**:
```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v
```

**Node.js**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Next Steps

After setup:

1. **Train the model** (if not already done)
2. **Evaluate the model**: `python src/training/evaluate.py --checkpoint ./checkpoints/best.pth`
3. **Export the model**: `python src/training/export_model.py --checkpoint ./checkpoints/best.pth --format both`
4. **Use the web interface** to classify images
5. **Integrate the API** into your applications

## Production Deployment

For production deployment:

1. **Build the frontend**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Use Docker** (see README.md for Docker instructions)

3. **Set up a production server** (Gunicorn, Nginx, etc.)

4. **Configure environment variables** for model path and API settings

---

For more details, see the main [README.md](README.md) file.

