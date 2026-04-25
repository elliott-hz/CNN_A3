import { useState, useRef, useEffect, useCallback } from 'react';
import { analyzeVideo } from '../services/api';
import './VideoResultsDisplay.css';

const FRAME_INTERVAL = 0.2; // 200ms = 5fps
const MAX_VIDEO_DURATION = 20; // seconds

const VideoResultsDisplay = ({ videoFile, isProcessing }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState('idle'); // idle, analyzing, complete, error
  const [progress, setProgress] = useState(0);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');

  // Load video when file is provided
  useEffect(() => {
    if (videoFile && videoRef.current) {
      const videoURL = URL.createObjectURL(videoFile);
      videoRef.current.src = videoURL;
      setVideoLoaded(true);
      setAnalysisStatus('idle');
      setProgress(0);
      setAnalysisResults(null);
      setErrorMessage('');

      // Auto-play video (may fail due to browser policy)
      videoRef.current.play().catch(err => {
        console.log('Autoplay blocked, user can click play button:', err.message);
        setIsPlaying(false);
      });

      return () => {
        URL.revokeObjectURL(videoURL);
      };
    }
  }, [videoFile]);

  // Analyze video when loaded
  useEffect(() => {
    if (videoLoaded && videoFile) {
      analyzeVideoFile();
    }
  }, [videoLoaded]);

  // Analyze the entire video file
  const analyzeVideoFile = async () => {
    setAnalysisStatus('analyzing');
    setProgress(0);
    setErrorMessage('');

    try {
      // Start analysis (this will take 1-2 minutes for 20s video)
      const results = await analyzeVideo(videoFile);
      
      if (results.success) {
        setAnalysisResults(results);
        setAnalysisStatus('complete');
        setProgress(100);
        console.log(`Video analysis complete: ${results.total_frames} frames analyzed`);
      } else {
        throw new Error('Analysis returned unsuccessful');
      }
    } catch (error) {
      console.error('Video analysis failed:', error);
      setAnalysisStatus('error');
      setErrorMessage(error.message || 'Failed to analyze video');
    }
  };

  // Linear interpolation between two values
  const lerp = (start, end, t) => {
    return start + (end - start) * t;
  };

  // Interpolate bounding box coordinates
  const lerpBbox = (bbox1, bbox2, t) => {
    return [
      lerp(bbox1[0], bbox2[0], t),
      lerp(bbox1[1], bbox2[1], t),
      lerp(bbox1[2], bbox2[2], t),
      lerp(bbox1[3], bbox2[3], t),
    ];
  };

  // Get emotion color for labels
  const getEmotionColor = (emotion) => {
    const colors = {
      happy: '#4CAF50',
      angry: '#f44336',
      relaxed: '#2196F3',
      frown: '#FF9800',
      alert: '#9C27B0'
    };
    return colors[emotion.toLowerCase()] || '#666';
  };

  // Get emotion emoji
  const getEmotionEmoji = (emotion) => {
    const emojis = {
      happy: '😊',
      angry: '😠',
      relaxed: '😌',
      frown: '😟',
      alert: '👀'
    };
    return emojis[emotion.toLowerCase()] || '🐕';
  };

  // Draw annotations with interpolation
  const drawAnnotations = useCallback(() => {
    if (!canvasRef.current || !videoRef.current || !analysisResults) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    if (!video || video.videoWidth === 0) return;

    // Set canvas size to match video display size
    canvas.width = video.offsetWidth;
    canvas.height = video.offsetHeight;

    // Clear previous annotations
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Find the two frames surrounding current time
    const frames = analysisResults.frames;
    const currentFrameIndex = Math.floor(currentTime / FRAME_INTERVAL);
    
    const frameBefore = frames[currentFrameIndex] || frames[frames.length - 1];
    const frameAfter = frames[currentFrameIndex + 1] || frames[frames.length - 1];

    if (!frameBefore || !frameAfter) return;

    // Calculate interpolation factor (0.0 to 1.0)
    const frameTimeBefore = currentFrameIndex * FRAME_INTERVAL;
    const frameTimeAfter = (currentFrameIndex + 1) * FRAME_INTERVAL;
    const t = (currentTime - frameTimeBefore) / (frameTimeAfter - frameTimeBefore);

    // Calculate scaling factors (video may be displayed at different size than actual resolution)
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    // Get detections from both frames
    const detectionsBefore = frameBefore.detections || [];
    const detectionsAfter = frameAfter.detections || [];

    // Match detections between frames and interpolate
    // Simple matching: use the same dog_id index
    const maxDogs = Math.max(detectionsBefore.length, detectionsAfter.length);

    for (let i = 0; i < maxDogs; i++) {
      const detBefore = detectionsBefore[i];
      const detAfter = detectionsAfter[i];

      // If detection exists in both frames, interpolate position
      if (detBefore && detAfter) {
        const interpolatedBbox = lerpBbox(
          detBefore.bbox,
          detAfter.bbox,
          Math.min(1, Math.max(0, t))
        );

        // Use emotion from the closer frame
        const useBefore = t < 0.5;
        const detection = useBefore ? detBefore : detAfter;

        drawDetection(ctx, interpolatedBbox, detection, scaleX, scaleY, canvas.width, canvas.height);
      } else if (detBefore) {
        // Only exists in before frame
        drawDetection(ctx, detBefore.bbox, detBefore, scaleX, scaleY, canvas.width, canvas.height);
      } else if (detAfter) {
        // Only exists in after frame
        drawDetection(ctx, detAfter.bbox, detAfter, scaleX, scaleY, canvas.width, canvas.height);
      }
    }
  }, [analysisResults, currentTime]);

  // Draw a single detection
  const drawDetection = (ctx, bbox, detection, scaleX, scaleY, canvasWidth, canvasHeight) => {
    const [x1, y1, x2, y2] = bbox;
    const color = getEmotionColor(detection.emotion);
    
    // Scale coordinates to canvas size
    const scaledX1 = x1 * scaleX;
    const scaledY1 = y1 * scaleY;
    const scaledX2 = x2 * scaleX;
    const scaledY2 = y2 * scaleY;
    const boxWidth = scaledX2 - scaledX1;
    const boxHeight = scaledY2 - scaledY1;

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(scaledX1, scaledY1, boxWidth, boxHeight);

    // Prepare emotion label
    const emoji = getEmotionEmoji(detection.emotion);
    const labelText = `${emoji} ${detection.emotion} (${(detection.emotion_confidence * 100).toFixed(1)}%)`;
    
    // Calculate text dimensions
    ctx.font = 'bold 16px Arial';
    const textMetrics = ctx.measureText(labelText);
    const textHeight = 20;
    const padding = 5;
    const labelWidth = textMetrics.width + padding * 2;
    const labelHeight = textHeight + padding * 2;

    // Smart positioning for emotion label (top of box)
    let labelY = scaledY1 - labelHeight;
    
    // If not enough space above, place inside the box
    if (labelY < 0) {
      labelY = scaledY1 + 5;
    }

    // Draw label background
    ctx.fillStyle = color;
    ctx.fillRect(scaledX1, labelY, labelWidth, labelHeight);

    // Draw label text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(labelText, scaledX1 + padding, labelY + textHeight + padding - 2);

    // Draw dog ID label (bottom of box)
    const idText = `Dog #${detection.dog_id + 1}`;
    ctx.font = 'bold 14px Arial';
    const idMetrics = ctx.measureText(idText);
    const idWidth = idMetrics.width + padding * 2;
    const idHeight = 18;

    let idY = scaledY2 + 5;
    
    // If not enough space below, place inside the box at bottom
    if (idY + idHeight > canvasHeight) {
      idY = scaledY2 - idHeight - 5;
    }

    // Draw ID background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(scaledX1, idY, idWidth, idHeight);
    
    // Draw ID text
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(idText, scaledX1 + padding, idY + 14);
  };

  // Update current time and redraw on video timeupdate
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !analysisResults) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    
    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [analysisResults]);

  // Redraw annotations when current time changes
  useEffect(() => {
    if (videoLoaded && analysisResults) {
      drawAnnotations();
    }
  }, [currentTime, analysisResults, videoLoaded, drawAnnotations]);

  // Redraw on window resize
  useEffect(() => {
    const handleResize = () => {
      if (videoLoaded && analysisResults) {
        drawAnnotations();
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [videoLoaded, analysisResults, drawAnnotations]);

  // Handle video play/pause
  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  if (!videoFile) {
    return null;
  }

  return (
    <div className="video-results-display">
      {/* Video Player Section */}
      <div className="video-section">
        <div className="video-container">
          <video
            ref={videoRef}
            className="video-player"
            playsInline
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
            onLoadedMetadata={() => {
              if (analysisResults) {
                setTimeout(drawAnnotations, 100);
              }
            }}
          />
          
          {/* Overlay canvas for annotations - positioned on top of video */}
          <canvas 
            ref={canvasRef} 
            className="annotation-overlay"
          />
          
          {/* Processing indicator overlay */}
          {analysisStatus === 'analyzing' && (
            <div className="processing-overlay">
              <div className="spinner"></div>
              <p>Analyzing video...</p>
              <p className="progress-text">{Math.round(progress)}% complete</p>
              <p className="hint">This may take 1-2 minutes for a 20-second video</p>
            </div>
          )}

          {/* Error overlay */}
          {analysisStatus === 'error' && (
            <div className="processing-overlay error">
              <p className="error-icon">⚠️</p>
              <p className="error-message">{errorMessage}</p>
            </div>
          )}
        </div>

        {/* Video Controls */}
        <div className="video-controls">
          <button 
            className="control-button play-pause"
            onClick={togglePlayPause}
            disabled={analysisStatus !== 'complete'}
          >
            {isPlaying ? '⏸️ Pause' : '▶️ Play'}
          </button>
          
          <div className="status-indicator">
            <span className={`status-dot ${analysisStatus}`}></span>
            <span className="status-text">
              {analysisStatus === 'idle' && 'Loading...'}
              {analysisStatus === 'analyzing' && 'Analyzing video...'}
              {analysisStatus === 'complete' && 'Ready to play'}
              {analysisStatus === 'error' && 'Analysis failed'}
            </span>
          </div>

          {analysisResults && (
            <div className="video-info">
              <span>{analysisResults.total_frames} frames analyzed</span>
              <span>•</span>
              <span>{analysisResults.video_duration.toFixed(1)}s video</span>
            </div>
          )}
        </div>
      </div>

      {/* Results Section */}
      {analysisStatus === 'complete' && analysisResults && (
        <div className="results-section">
          {analysisResults.frames.length === 0 ? (
            <div className="no-detection">
              <p>🎥 No dogs detected in video</p>
              <p className="hint">Try uploading a video with clearer dog faces</p>
            </div>
          ) : (
            <div className="analysis-summary">
              <p className="summary-title">✅ Video Analysis Complete</p>
              <p className="summary-text">
                Analyzed {analysisResults.total_frames} frames over {analysisResults.video_duration.toFixed(1)} seconds
              </p>
              <p className="hint">Play the video to see smooth bounding box animations</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VideoResultsDisplay;
