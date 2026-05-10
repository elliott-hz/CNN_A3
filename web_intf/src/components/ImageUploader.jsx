import { useState, useRef } from 'react';
import { detectEmotion } from '../services/api';
import './ImageUploader.css';

const MAX_IMAGE_SIZE = 1280; // Maximum dimension for inference (increased for better label visibility)

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

  // Resize image to max dimension while maintaining aspect ratio (only downscale, never upscale)
  const resizeImage = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          let width = img.width;
          let height = img.height;

          // Only resize if image exceeds MAX_IMAGE_SIZE
          if (width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE) {
            // Calculate new dimensions - only downscale
            if (width > height) {
              height = Math.round((height * MAX_IMAGE_SIZE) / width);
              width = MAX_IMAGE_SIZE;
            } else {
              width = Math.round((width * MAX_IMAGE_SIZE) / height);
              height = MAX_IMAGE_SIZE;
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
          } else {
            // Image is already within limits, use original
            resolve(file);
          }
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
    setIsLoading(true);
    
    try {
      // Resize image if needed
      const resizedBlob = await resizeImage(file);
      const resizedFile = new File([resizedBlob], file.name, {
        type: file.type,
        lastModified: Date.now()
      });
      
      setSelectedImage(resizedFile);
      
      // Automatically start detection after image is loaded
      const reader = new FileReader();
      reader.onloadend = async () => {
        try {
          const results = await detectEmotion(resizedFile);
          onResults(results, reader.result);
        } catch (err) {
          setError(err.message);
          console.error('Detection error:', err);
        } finally {
          setIsLoading(false);
        }
      };
      reader.readAsDataURL(resizedBlob);
    } catch (err) {
      console.error('Error resizing image:', err);
      setError('Failed to process image');
      setIsLoading(false);
    }
  };

  return (
    <div className="image-uploader">
      {/* <h2>🐕 Dog Emotion Recognition</h2> */}
      
      {/* Upload Area - Compact inline layout with auto-detection */}
      <div
        className={`upload-area ${isLoading ? 'loading' : ''} ${selectedImage ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => !isLoading && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        <div className="upload-placeholder">
          {isLoading ? (
            <>
              <div className="spinner loading-spinner"></div>
              <div>
                <p>Analyzing image...</p>
                <p className="upload-hint">Please wait while we detect emotions</p>
              </div>
            </>
          ) : (
            <>
              <div className="upload-icon">📷</div>
              <div>
                <p>{selectedImage ? '✅ Analysis complete! Upload another image to continue' : 'Click or drag & drop an image'}</p>
                {selectedImage && (
                  <p className="upload-hint">
                    📁 {selectedImage.name} • {(selectedImage.size / 1024).toFixed(1)} KB
                  </p>
                )}
                {!selectedImage && (
                  <p className="upload-hint">Supports JPEG, PNG (max 10MB, auto-resized to 640px)</p>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
