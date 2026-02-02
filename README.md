# Image Classifier Suite

A complete end-to-end image classification system that trains a deep CNN on CIFAR-10 and provides both a REST API and a beautiful React web interface for real-time image classification.

## 🎯 Project Overview

This project implements a production-ready image classification service with:
- **Deep Learning Model**: Enhanced ResNet-like CNN architecture achieving 94.28% accuracy on CIFAR-10
- **Training Pipeline**: Complete training system with data augmentation, learning rate scheduling, and checkpointing
- **REST API**: FastAPI-based inference service with CORS support
- **Web Interface**: Modern React frontend with drag-and-drop image upload
- **Model Export**: Support for TorchScript and ONNX export for deployment

## 📊 Results

- **Test Accuracy**: 94.28%
- **Model Parameters**: 12.7M parameters
- **Training Time**: ~30 epochs (configurable)
- **Inference Speed**: Real-time on Apple Silicon (MPS)

## 🏗️ Architecture

### Model Architecture

The CNN uses a ResNet-like architecture with residual blocks:

- **Initial Layer**: 3x3 convolution with 64 channels, BatchNorm, ReLU
- **Residual Blocks**: 4 layers with increasing depth:
  - Layer 1: 3 blocks, 64 channels
  - Layer 2: 3 blocks, 128 channels (downsampling)
  - Layer 3: 3 blocks, 256 channels (downsampling)
  - Layer 4: 2 blocks, 512 channels (downsampling)
- **Global Average Pooling**: Adaptive average pooling to 1x1
- **Dropout**: 0.3 dropout rate for regularization
- **Classification Head**: Fully connected layer (512 → 10 classes)

**Total Parameters**: 12,724,042

### Training Features

- **Data Augmentation**:
  - Random crop with padding (4px)
  - Random horizontal flip
  - Color jitter (brightness, contrast, saturation, hue)
  - Random rotation (±5 degrees)
  - Random erasing (Cutout-like, 30% probability)

- **Optimization**:
  - SGD optimizer with Nesterov momentum (0.9)
  - Learning rate: 0.1 (configurable)
  - Weight decay: 5e-4
  - Gradient clipping (max norm: 1.0)

- **Learning Rate Scheduling**:
  - Cosine annealing with warmup (5 epochs default)
  - Alternative: Step LR scheduler

- **Regularization**:
  - Label smoothing (0.1 default)
  - Dropout (0.3 default)
  - Weight initialization: Kaiming initialization

- **Checkpointing**:
  - Saves best model based on validation accuracy
  - Saves latest checkpoint every epoch
  - Resume training from checkpoint supported

## 📁 Project Structure

```
imageclassifier-suite/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # CIFAR-10 data loading with augmentation
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py              # ResNet-like CNN model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py            # Training script
│   │   ├── evaluate.py         # Evaluation with confusion matrix
│   │   └── export_model.py     # Model export (TorchScript/ONNX)
│   └── api/
│       ├── __init__.py
│       └── main.py             # FastAPI inference service
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.js
│   │   │   └── PredictionResult.js
│   │   ├── App.js
│   │   └── index.js
│   ├── public/
│   └── package.json
├── checkpoints/               # Model checkpoints (gitignored)
├── outputs/                   # Evaluation outputs (gitignored)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── setup.sh                   # Setup script
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 16+ and npm
- PyTorch (with MPS support for Apple Silicon or CUDA for NVIDIA GPUs)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/VenkataVardineni/imageclassifier-suite.git
cd imageclassifier-suite
```

2. **Run the setup script**:
```bash
chmod +x setup.sh
./setup.sh
```

Or follow the manual setup in [SETUP.md](SETUP.md).

3. **Train the model** (if not using pre-trained weights):
```bash
python src/training/train.py --epochs 100 --batch-size 128
```

4. **Start the backend API**:
```bash
python src/api/main.py
```

5. **Start the frontend** (in a new terminal):
```bash
cd frontend
npm install
npm start
```

6. **Open your browser**: http://localhost:3000

## 📚 Detailed Documentation

### Training

Train the model with customizable hyperparameters:

```bash
python src/training/train.py \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.1 \
  --scheduler cosine_warmup \
  --warmup-epochs 5 \
  --label-smoothing 0.1 \
  --dropout 0.3 \
  --checkpoint-dir ./checkpoints
```

**Key Arguments**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--scheduler`: LR scheduler type: `cosine_warmup`, `cosine`, or `step` (default: cosine_warmup)
- `--warmup-epochs`: Number of warmup epochs (default: 5)
- `--label-smoothing`: Label smoothing factor (default: 0.1)
- `--dropout`: Dropout rate (default: 0.3)
- `--resume`: Path to checkpoint to resume from

### Evaluation

Evaluate the trained model and generate a confusion matrix:

```bash
python src/training/evaluate.py \
  --checkpoint ./checkpoints/best.pth \
  --output-dir ./outputs
```

This will:
- Compute test accuracy
- Generate classification report
- Create a confusion matrix visualization

### Model Export

Export the model for deployment:

```bash
# Export to TorchScript
python src/training/export_model.py \
  --checkpoint ./checkpoints/best.pth \
  --format torchscript

# Export to ONNX
python src/training/export_model.py \
  --checkpoint ./checkpoints/best.pth \
  --format onnx

# Export to both
python src/training/export_model.py \
  --checkpoint ./checkpoints/best.pth \
  --format both
```

### API Endpoints

The FastAPI service provides:

- `GET /`: API information
- `GET /health`: Health check and model status
- `POST /predict`: Upload an image and get classification

**Example API Usage**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

**Response**:
```json
{
  "predicted_class": "airplane",
  "confidence": 0.95,
  "top3_predictions": [
    {"class": "airplane", "confidence": 0.95},
    {"class": "bird", "confidence": 0.03},
    {"class": "ship", "confidence": 0.02}
  ]
}
```

## 🎨 Frontend Features

The React frontend provides:

- **Drag & Drop Upload**: Intuitive image upload interface
- **Real-time Predictions**: Instant classification results
- **Visual Feedback**: 
  - Confidence bars with color coding
  - Top 3 predictions display
  - Class emojis for easy identification
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: Clear error messages

## 📦 Dependencies

### Python Dependencies

See `requirements.txt` for complete list. Key libraries:

- `torch>=2.0.0`: Deep learning framework
- `torchvision>=0.15.0`: Computer vision utilities
- `fastapi>=0.104.0`: Web framework
- `uvicorn[standard]>=0.24.0`: ASGI server
- `pillow>=10.0.0`: Image processing
- `numpy>=1.24.0`: Numerical computing
- `matplotlib>=3.7.0`: Plotting
- `seaborn>=0.12.0`: Statistical visualization
- `scikit-learn>=1.3.0`: Machine learning utilities
- `onnx>=1.15.0`: ONNX model format
- `onnxruntime>=1.16.0`: ONNX inference

### Frontend Dependencies

- `react>=18.2.0`: UI framework
- `react-dom>=18.2.0`: React DOM rendering
- `react-scripts>=5.0.1`: Create React App scripts
- `axios>=1.6.0`: HTTP client

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t imageclassifier-suite .

# Run the container
docker run -p 8000:8000 imageclassifier-suite
```

**Note**: Ensure model checkpoints are in the `checkpoints/` directory before building.

## 📈 Training Results

### Model Performance

After training for 30 epochs with the enhanced architecture:

- **Training Accuracy**: 95.71%
- **Validation Accuracy**: 94.28%
- **Training Loss**: 0.6038
- **Validation Loss**: 0.6366

### Training Configuration Used

- **Epochs**: 30 (can be extended to 100 for better results)
- **Batch Size**: 128
- **Learning Rate**: 0.1 with cosine warmup
- **Warmup Epochs**: 3
- **Optimizer**: SGD with Nesterov momentum
- **Data Augmentation**: Full augmentation pipeline enabled

## 🎯 CIFAR-10 Classes

The model classifies images into 10 categories:

1. ✈️ Airplane
2. 🚗 Automobile
3. 🐦 Bird
4. 🐱 Cat
5. 🦌 Deer
6. 🐶 Dog
7. 🐸 Frog
8. 🐴 Horse
9. 🚢 Ship
10. 🚚 Truck

## 🔧 Configuration

### Model Configuration

Model parameters can be adjusted in `src/models/cnn.py`:
- Number of residual blocks per layer
- Channel widths
- Dropout rate

### Training Configuration

All training hyperparameters are configurable via command-line arguments in `src/training/train.py`.

### API Configuration

API settings in `src/api/main.py`:
- Model path (via `MODEL_PATH` environment variable)
- Host and port (via command-line arguments)

## 📝 Development

### Running Tests

```bash
# Test model creation
python -c "from src.models.cnn import create_model; import torch; model = create_model(device='cpu'); print('Model OK')"

# Test data loading
python -c "from src.data.dataset import get_cifar10_dataloaders; train, test = get_cifar10_dataloaders(batch_size=32); print('Data OK')"
```

### Code Structure

- **Modular Design**: Each component (data, models, training, API) is self-contained
- **Type Hints**: Python type hints for better code clarity
- **Error Handling**: Comprehensive error handling throughout
- **Documentation**: Docstrings for all functions and classes

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- PyTorch team for the excellent framework
- FastAPI for the modern web framework
- React team for the UI library

---

**Repository**: https://github.com/VenkataVardineni/imageclassifier-suite

**Author**: Venkata Vardineni

**Last Updated**: February 2025
