# Image Classifier Suite

Trains a CNN on CIFAR-10 and exposes inference API.

## Project Structure

```
imageclassifier-suite/
├── src/
│   ├── data/          # Data loading and preprocessing utilities
│   ├── models/        # CNN model definitions
│   ├── training/      # Training scripts
│   └── api/           # FastAPI inference service
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker configuration for deployment
└── README.md          # This file
```

## Features

- **CNN Model**: Small ResNet-like architecture for CIFAR-10 classification
- **Training**: Complete training pipeline with data augmentation, optimizer, LR scheduler, and checkpointing
- **Evaluation**: Test accuracy computation and confusion matrix generation
- **Model Export**: Export to TorchScript/ONNX for efficient deployment
- **Inference API**: FastAPI service for image classification

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python src/training/train.py
```

3. Evaluate the model:
```bash
python src/training/evaluate.py
```

4. Export the model:
```bash
python src/training/export_model.py
```

5. Run the inference API:
```bash
python src/api/main.py
```

Or using Docker:
```bash
docker build -t imageclassifier-suite .
docker run -p 8000:8000 imageclassifier-suite
```

## API Usage

Send a POST request to `/predict` with an image file:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

## CIFAR-10 Classes

The model predicts one of 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

