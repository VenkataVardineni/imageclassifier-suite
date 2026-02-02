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
    try:
        # Resize to 32x32 (CIFAR-10 input size)
        try:
            image = image.resize((32, 32), Image.Resampling.LANCZOS)
        except AttributeError:
            # Fallback for older PIL versions
            image = image.resize((32, 32), Image.LANCZOS)
        
        # Convert to RGB if needed (handles RGBA, L, P, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        # Explicitly use float32 to avoid MPS float64 issues
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Ensure the array has the right shape (H, W, C)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {img_array.shape}. Expected (H, W, 3)")
        
        # Normalize using CIFAR-10 statistics (explicitly float32)
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Ensure array is still float32 after operations
        img_array = img_array.astype(np.float32)
        
        # Convert to tensor and add batch dimension
        # Shape: (H, W, C) -> (C, H, W) -> (1, C, H, W)
        # Explicitly specify dtype as float32
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        
        # Ensure tensor is on the correct device
        img_tensor = img_tensor.to(device)
        
        return img_tensor
    except Exception as e:
        raise ValueError(f"Error in image preprocessing: {str(e)}")


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
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Try to open the image
        try:
            image = Image.open(BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file. Please upload a valid image (JPEG, PNG, etc.). Error: {str(e)}")
        
        # Verify image was loaded
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Preprocess image
        try:
            img_tensor = preprocess_image(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")
        
        # Run inference
        try:
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")
        
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
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"Unexpected error: {error_trace}")
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
