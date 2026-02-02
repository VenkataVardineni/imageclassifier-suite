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

6. (Optional) Start the React frontend:
```bash
cd frontend
npm install
npm start
```

Or using Docker:
```bash
docker build -t imageclassifier-suite .
docker run -p 8000:8000 imageclassifier-suite
```

## Frontend UI

A beautiful React frontend is available for easy interaction with the model.

### Setup Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The UI will open at http://localhost:3000

### Features

- 🖼️ Drag and drop image upload
- 📊 Real-time prediction display
- 🎯 Top 3 predictions with confidence scores
- 📱 Responsive design
- ✨ Beautiful, modern UI

## API Usage

### Using the Web UI

Simply open http://localhost:3000 and upload an image through the interface.

### Using cURL

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

