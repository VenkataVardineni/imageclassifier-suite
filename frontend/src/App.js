import React, { useState } from 'react';
import './App.css';
import ImageUploader from './components/ImageUploader';
import PredictionResult from './components/PredictionResult';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handlePrediction = async (imageFile) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedImage(reader.result);
    };
    reader.readAsDataURL(imageFile);

    // Send to API
    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Make sure the API server is running.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPrediction(null);
    setUploadedImage(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🖼️ Image Classifier Suite</h1>
          <p className="subtitle">CIFAR-10 Image Classification</p>
        </header>

        <div className="main-content">
          <div className="upload-section">
            <ImageUploader
              onImageUpload={handlePrediction}
              onReset={handleReset}
              loading={loading}
              hasResult={prediction !== null}
            />
          </div>

          {uploadedImage && (
            <div className="image-preview-section">
              <h2>Uploaded Image</h2>
              <div className="image-preview">
                <img src={uploadedImage} alt="Uploaded" />
              </div>
            </div>
          )}

          {loading && (
            <div className="loading-section">
              <div className="spinner"></div>
              <p>Analyzing image...</p>
            </div>
          )}

          {error && (
            <div className="error-section">
              <p className="error-message">❌ {error}</p>
            </div>
          )}

          {prediction && (
            <PredictionResult prediction={prediction} />
          )}
        </div>

        <footer className="footer">
          <p>Upload an image to classify it into one of 10 CIFAR-10 categories</p>
          <p className="categories">
            Categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;

