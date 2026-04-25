import { useState, useRef } from 'react';
import './VideoUploader.css';

const VideoUploader = ({ onVideoSelect }) => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      validateAndSetVideo(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      validateAndSetVideo(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const validateAndSetVideo = (file) => {
    // Validate file type
    if (!file.type.startsWith('video/')) {
      setError('Please select a video file');
      return;
    }

    // Validate file size (max 50MB)
    if (file.size > 50 * 1024 * 1024) {
      setError('Video size must be less than 50MB');
      return;
    }

    setError(null);
    setIsLoading(true);
    
    try {
      setSelectedVideo(file);
      
      // Notify parent component
      onVideoSelect(file);
      setIsLoading(false);
    } catch (err) {
      console.error('Error processing video:', err);
      setError('Failed to process video');
      setIsLoading(false);
    }
  };

  const clearVideo = () => {
    setSelectedVideo(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onVideoSelect(null);
  };

  return (
    <div className="video-uploader">
      {/* Upload Area */}
      <div
        className={`upload-area ${isLoading ? 'loading' : ''} ${selectedVideo ? 'has-video' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => !isLoading && !selectedVideo && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        
        <div className="upload-placeholder">
          {isLoading ? (
            <>
              <div className="spinner loading-spinner"></div>
              <div>
                <p>Processing video...</p>
                <p className="upload-hint">Please wait</p>
              </div>
            </>
          ) : selectedVideo ? (
            <>
              <div className="upload-icon">🎥</div>
              <div className="video-info">
                <p className="success-message">✅ Video loaded successfully!</p>
                <p className="upload-hint">
                  📁 {selectedVideo.name} • {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB
                </p>
                <button 
                  className="change-video-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearVideo();
                  }}
                >
                  🔄 Change Video
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="upload-icon">🎬</div>
              <div>
                <p>Click or drag & drop a video</p>
                <p className="upload-hint">Supports MP4, WebM, AVI (max 50MB)</p>
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

export default VideoUploader;
