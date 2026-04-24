import { useState, useRef } from 'react';
import { detectEmotion } from '../services/api';
import './ImageUploader.css';

const MAX_IMAGE_SIZE = 640; // Maximum dimension for inference

const ImageUploader = ({ onResults }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      validateAndSetImage(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      validateAndSetImage(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Resize image to max dimension while maintaining aspect ratio
  const resizeImage = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          let width = img.width;
          let height = img.height;

          // Calculate new dimensions
          if (width > height) {
            if (width > MAX_IMAGE_SIZE) {
              height = Math.round((height * MAX_IMAGE_SIZE) / width);
              width = MAX_IMAGE_SIZE;
            }
          } else {
            if (height > MAX_IMAGE_SIZE) {
              width = Math.round((width * MAX_IMAGE_SIZE) / height);
              height = MAX_IMAGE_SIZE;
            }
          }

          // Create canvas and resize
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);

          // Convert to blob
          canvas.toBlob((blob) => {
            resolve(blob);
          }, file.type, 0.9); // 90% quality
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  };

  const validateAndSetImage = async (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('Image size must be less than 10MB');
      return;
    }

    setError(null);
    
    try {
      // Resize image if needed
      const resizedBlob = await resizeImage(file);
      const resizedFile = new File([resizedBlob], file.name, {
        type: file.type,
        lastModified: Date.now()
      });
      
      setSelectedImage(resizedFile);
    } catch (err) {
      console.error('Error resizing image:', err);
      setError('Failed to process image');
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const results = await detectEmotion(selectedImage);
      
      // Create preview URL for the annotated display only after successful detection
      const reader = new FileReader();
      reader.onloadend = () => {
        onResults(results, reader.result);
      };
      reader.readAsDataURL(selectedImage);
    } catch (err) {
      setError(err.message);
      console.error('Detection error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setError(null);
    onResults(null, null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="image-uploader">
      
      {/* Upload Area - Compact inline layout */}
      <div
        className={`upload-area ${selectedImage ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        <div className="upload-placeholder">
          <div className="upload-icon">📷</div>
          <div>
            <p>{selectedImage ? '✅ Image ready for analysis' : 'Click or drag & drop an image'}</p>
            {selectedImage && (
              <p className="upload-hint">
                📁 {selectedImage.name} • {(selectedImage.size / 1024).toFixed(1)} KB
              </p>
            )}
            {!selectedImage && (
              <p className="upload-hint">Supports JPEG, PNG (max 10MB, auto-resized to 640px)</p>
            )}
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Action Buttons */}
      {selectedImage && (
        <div className="action-buttons">
          <button 
            className="btn btn-primary" 
            onClick={handleUpload}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <div className="spinner"></div>
                Analyzing...
              </>
            ) : (
              '🚀 Detect Emotion'
            )}
          </button>
          <button 
            className="btn btn-secondary" 
            onClick={handleReset}
            disabled={isLoading}
          >
            🔄 Reset
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
