import { useState, useRef, useEffect } from 'react';
import './LiveStream.css';

const LiveStream = () => {
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const streamRef = useRef(null);

  useEffect(() => {
    startCamera();
    
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Unable to access camera. Please check permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
  };

  return (
    <div className="live-stream">
      <div className="live-video-container">
        <video 
          ref={videoRef}
          autoPlay 
          playsInline
          muted
          className="live-video"
        />
        
        {error && (
          <div className="error-overlay">
            <p>⚠️ {error}</p>
          </div>
        )}
        
        <div className="live-indicator">
          <span className="live-dot"></span>
          LIVE STREAM
        </div>
        
        <div className="live-info">
          <p>📹 Camera active • Real-time video feed</p>
          <p className="hint">Future: Emotion detection will be added here</p>
        </div>
      </div>
    </div>
  );
};

export default LiveStream;
