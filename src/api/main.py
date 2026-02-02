"""
FastAPI inference service for CIFAR-10 image classification.
"""

import os
import sys
from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.cnn import create_model
from data.dataset import CIFAR10_CLASSES


app = FastAPI(title="Image Classifier Suite", version="1.0.0")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None


def load_model(model_path: str):
    """Load the trained model."""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = create_model(num_classes=10, device=device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f'Model loaded from {model_path}')


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference."""
    # Resize to 32x32 (CIFAR-10 input size)
    image = image.resize((32, 32))
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using CIFAR-10 statistics
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    img_array = (img_array - mean) / std
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    model_path = os.getenv('MODEL_PATH', './checkpoints/best.pth')
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        print(f'Warning: Model checkpoint not found at {model_path}')
        print('Please train the model first or set MODEL_PATH environment variable')


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Image Classifier Suite API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload an image to get classification",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON response with predicted class and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Preprocess image
        img_tensor = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        
        # Format results
        result = {
            "predicted_class": CIFAR10_CLASSES[predicted.item()],
            "confidence": float(confidence.item()),
            "top3_predictions": [
                {
                    "class": CIFAR10_CLASSES[idx.item()],
                    "confidence": float(prob.item())
                }
                for prob, idx in zip(top3_probs[0], top3_indices[0])
            ]
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FastAPI inference server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--model-path', type=str, default='./checkpoints/best.pth', help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Set model path environment variable
    os.environ['MODEL_PATH'] = args.model_path
    
    uvicorn.run(app, host=args.host, port=args.port)
