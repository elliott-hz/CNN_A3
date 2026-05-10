import { useState, useRef, useEffect } from 'react';
import './LiveStream.css';

const LiveStream = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [deviceInfo, setDeviceInfo] = useState(null);
  const [fps, setFps] = useState(0);
  const [detections, setDetections] = useState([]);
  const [latestTimestamp, setLatestTimestamp] = useState(0);
  const detectionsRef = useRef([]);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const frameIntervalRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);

  useEffect(() => {
    checkDeviceSupport();
    startCamera();
    
    return () => {
      stopCamera();
      disconnectWebSocket();
    };
  }, []);

  // Check backend device support
  const checkDeviceSupport = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      const data = await response.json();
      setDeviceInfo(data.device || 'Unknown');
      
      // Warn if running on CPU
      if (data.device === 'CPU') {
        setError('⚠️ CPU DETECTED: Live streaming requires GPU (MPS or CUDA). Performance will be very slow. Please use a Mac with Apple Silicon or NVIDIA GPU.');
      }
    } catch (err) {
      console.error('Error checking device:', err);
      setError('Cannot connect to backend server. Please ensure it\'s running.');
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 }
        },
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
        
        // Start sending frames after video is ready
        videoRef.current.onloadedmetadata = () => {
          connectWebSocket();
          startFrameCapture();
        };
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      setError('Unable to access camera. Please check permissions.');
    }
  };

  const connectWebSocket = () => {
    const wsUrl = 'ws://localhost:8000/ws/live-stream';
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('✅ WebSocket connected');
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.success) {
          // Only update if this is a newer result
          const frameTimestamp = data.timestamp || 0;
          if (frameTimestamp >= latestTimestamp) {
            setDetections(data.detections || []);
            setLatestTimestamp(frameTimestamp);
          }
          
          // Calculate FPS based on received frames
          frameCountRef.current += 1;
          const now = Date.now();
          if (now - lastFrameTimeRef.current >= 1000) {
            setFps(frameCountRef.current);
            frameCountRef.current = 0;
            lastFrameTimeRef.current = now;
          }
        } else {
          console.error('Inference error:', data.error);
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };
    
    wsRef.current.onerror = (err) => {
      console.error('WebSocket error:', err);
      setError('WebSocket connection failed. Check if backend is running.');
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket closed');
    };
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  const startFrameCapture = () => {
    // OPTIMIZATION: Capture frames at higher rate for smoother experience
    // MPS can handle ~10 FPS, so we send every 100ms instead of 200ms
    const captureInterval = deviceInfo?.includes('MPS') ? 100 : 
                           deviceInfo?.includes('CUDA') ? 50 : 1000;
    
    frameIntervalRef.current = setInterval(() => {
      if (!canvasRef.current || !videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        return;
      }
      
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      
      // Set canvas size to match video (only when needed)
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      
      // Draw current video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Convert to base64 with lower quality for faster transmission (0.6 instead of 0.8)
      const base64Image = canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
      
      // Send to backend - non-blocking, don't wait for response
      wsRef.current.send(JSON.stringify({
        frame: base64Image,
        timestamp: Date.now()
      }));
    }, captureInterval);
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setIsStreaming(false);
    setFps(0);
    setDetections([]);
  };

  const drawDetections = () => {
    if (!canvasRef.current || !videoRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw current video frame
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    // Draw bounding boxes and labels using ref to avoid stale closures in animation loop
    detectionsRef.current.forEach((det, idx) => {
      const [x1, y1, x2, y2] = det.bbox;
      const width = x2 - x1;
      const height = y2 - y1;
      
      // Color based on emotion - MATCHES BACKEND EMOTION CLASSES
      const colors = {
        'happy': '#4CAF50',      // Green
        'angry': '#f44336',      // Red
        'relaxed': '#2196F3',    // Blue
        'frown': '#FF9800',      // Orange
        'alert': '#9C27B0'       // Purple
      };
      const color = colors[det.emotion] || '#666666';  // Default to gray instead of white
      
      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, width, height);
      
      // Draw label background
      const label = `${det.emotion} ${(det.emotion_confidence * 100).toFixed(1)}%`;
      ctx.font = 'bold 16px Arial';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
      
      // Draw label text - ALWAYS WHITE for contrast
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x1 + 5, y1 - 7);
    });
  };

  // Keep detectionsRef in sync with detections state
  useEffect(() => {
    detectionsRef.current = detections;
  }, [detections]);

  // Redraw canvas continuously for smooth video (60 FPS)
  useEffect(() => {
    if (!isStreaming) return;
    
    const render = () => {
      drawDetections();
      animationFrameRef.current = requestAnimationFrame(render);
    };
    
    animationFrameRef.current = requestAnimationFrame(render);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [isStreaming]);

  return (
    <div className="live-stream">
      <div className="device-info">
        <span className={`device-badge ${deviceInfo?.includes('CPU') ? 'cpu-warning' : 'gpu-ok'}`}>
          {deviceInfo || 'Checking...'}
        </span>
        {fps > 0 && <span className="fps-counter">{fps} FPS</span>}
      </div>
      
      <div className="live-video-container">
        <video 
          ref={videoRef}
          autoPlay 
          playsInline
          muted
          className="live-video"
          style={{ display: 'none' }}
        />
        
        <canvas 
          ref={canvasRef}
          className="live-video"
        />
        
        {error && deviceInfo === 'CPU' && (
          <div className="error-overlay cpu-error">
            <div className="error-content">
              <h3>⚠️ CPU Not Supported for Live Streaming</h3>
              <p>{error}</p>
              <div className="recommendations">
                <h4>Recommended Solutions:</h4>
                <ul>
                  <li>✅ Use iMac with Apple Silicon (M1/M2/M3) - MPS GPU acceleration</li>
                  <li>✅ Use computer with NVIDIA GPU - CUDA acceleration</li>
                  <li>❌ Intel MacBook CPU-only inference is too slow for real-time</li>
                </ul>
              </div>
            </div>
          </div>
        )}
        
        {error && deviceInfo !== 'CPU' && (
          <div className="error-overlay">
            <p>⚠️ {error}</p>
          </div>
        )}
        
        <div className="live-indicator">
          <span className="live-dot"></span>
          LIVE STREAM
        </div>
        
        <div className="live-info">
          <p>📹 Camera active • Real-time emotion detection</p>
          {deviceInfo?.includes('MPS') && (
            <p className="performance-hint">💡 Apple Silicon detected - Optimized for 10 FPS (frame skipping enabled)</p>
          )}
          {deviceInfo?.includes('CUDA') && (
            <p className="performance-hint">💡 NVIDIA GPU detected - Optimized for 20 FPS (frame skipping enabled)</p>
          )}
          {fps > 0 && (
            <p className="performance-hint">⚡ Actual processing rate: {fps} FPS</p>
          )}
        </div>
      </div>
      
      <div className="controls">
        <button 
          className={isStreaming ? 'btn-stop' : 'btn-start'}
          onClick={isStreaming ? stopCamera : startCamera}
        >
          {isStreaming ? '⏹ Stop Stream' : '▶ Start Stream'}
        </button>
      </div>
    </div>
  );
};

export default LiveStream;