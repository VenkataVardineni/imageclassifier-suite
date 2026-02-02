import React, { useRef } from 'react';
import './ImageUploader.css';

function ImageUploader({ onImageUpload, onReset, loading, hasResult }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
      }
      onImageUpload(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file);
    } else {
      alert('Please drop an image file');
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="image-uploader">
      <div
        className={`upload-area ${loading ? 'loading' : ''} ${hasResult ? 'has-result' : ''}`}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          disabled={loading}
        />
        
        {!hasResult && !loading && (
          <>
            <div className="upload-icon">📤</div>
            <p className="upload-text">Click or drag an image here to upload</p>
            <p className="upload-hint">Supports: JPG, PNG, GIF, etc.</p>
          </>
        )}
        
        {loading && (
          <>
            <div className="upload-icon">⏳</div>
            <p className="upload-text">Processing...</p>
          </>
        )}
        
        {hasResult && !loading && (
          <>
            <div className="upload-icon">✅</div>
            <p className="upload-text">Image uploaded successfully</p>
            <button className="reset-button" onClick={(e) => { e.stopPropagation(); onReset(); }}>
              Upload Another Image
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default ImageUploader;

