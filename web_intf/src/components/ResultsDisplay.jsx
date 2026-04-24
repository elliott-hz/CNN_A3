import { useState, useRef, useEffect } from 'react';
import './ResultsDisplay.css';

const ResultsDisplay = ({ results, imagePreview }) => {
  const canvasRef = useRef(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Draw bounding boxes and labels on canvas when results change
  useEffect(() => {
    if (!results || !results.success || !imagePreview || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the original image
      ctx.drawImage(img, 0, 0);

      // Draw bounding boxes and labels for each detection
      results.results.forEach((detection, index) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const color = getEmotionColor(detection.emotion);
        
        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Prepare label text
        const emoji = getEmotionEmoji(detection.emotion);
        const labelText = `${emoji} ${detection.emotion} (${(detection.emotion_confidence * 100).toFixed(1)}%)`;
        
        // Calculate text dimensions for background
        ctx.font = 'bold 16px Arial';
        const textMetrics = ctx.measureText(labelText);
        const textHeight = 20;
        const padding = 5;
        const labelWidth = textMetrics.width + padding * 2;
        const labelHeight = textHeight + padding * 2;

        // Smart positioning for top label (emotion)
        let labelY = y1 - labelHeight;
        let labelInsideTop = false;
        
        // If not enough space above, place inside the box at top
        if (labelY < 0) {
          labelY = y1 + 5; // Small offset from top edge
          labelInsideTop = true;
        }

        // Draw label background
        ctx.fillStyle = color;
        ctx.fillRect(x1, labelY, labelWidth, labelHeight);

        // Draw label text
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(labelText, x1 + padding, labelY + textHeight + padding - 2);

        // Smart positioning for bottom label (dog ID)
        const idText = `Dog #${detection.dog_id + 1}`;
        ctx.font = 'bold 14px Arial';
        const idMetrics = ctx.measureText(idText);
        const idWidth = idMetrics.width + padding * 2;
        const idHeight = textHeight;

        let idY = y2 + padding;
        let idInsideBottom = false;
        
        // If not enough space below, place inside the box at bottom
        if (idY + idHeight > canvas.height) {
          idY = y2 - idHeight - 5; // Small offset from bottom edge
          idInsideBottom = true;
        }

        // Draw ID background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x1, idY, idWidth, idHeight);
        
        // Draw ID text
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(idText, x1 + padding, idY + 14);
      });

      setImageLoaded(true);
    };

    img.src = imagePreview;
  }, [results, imagePreview]);

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

  if (!results || !results.success) {
    return null;
  }

  return (
    <div className="results-display">
      <h3>🎯 Detection Results</h3>
      
      {results.results.length === 0 ? (
        <div className="no-detection">
          <p>😕 No dogs detected in this image</p>
          <p className="hint">Try uploading a clearer image with visible dog faces</p>
        </div>
      ) : (
        <>
          <p className="result-summary">{results.message}</p>
          
          {/* Annotated Image with Bounding Boxes */}
          {imagePreview && (
            <div className="annotated-image-container">
              <canvas 
                ref={canvasRef}
                className="annotated-canvas"
              />
              {!imageLoaded && (
                <div className="canvas-loading">
                  <div className="spinner"></div>
                  <p>Drawing annotations...</p>
                </div>
              )}
            </div>
          )}
          
          <div className="detections-grid">
            {results.results.map((detection, index) => (
              <div key={detection.dog_id} className="detection-card">
                <div className="detection-header">
                  <span className="dog-id">Dog #{detection.dog_id + 1}</span>
                  <span 
                    className="emotion-badge"
                    style={{ backgroundColor: getEmotionColor(detection.emotion) }}
                  >
                    {getEmotionEmoji(detection.emotion)} {detection.emotion}
                  </span>
                </div>
                
                <div className="detection-details">
                  <div className="detail-row">
                    <span className="label">Detection Confidence:</span>
                    <span className="value">
                      {(detection.detection_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="detail-row">
                    <span className="label">Emotion Confidence:</span>
                    <span className="value">
                      {(detection.emotion_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="detail-row">
                    <span className="label">Bounding Box:</span>
                    <span className="value bbox">
                      [{detection.bbox.map(v => Math.round(v)).join(', ')}]
                    </span>
                  </div>
                </div>
                
                <div className="probability-bars">
                  <h4>Emotion Probabilities:</h4>
                  {Object.entries(detection.emotion_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([emotion, prob]) => (
                      <div key={emotion} className="prob-bar">
                        <div className="prob-label">
                          <span>{getEmotionEmoji(emotion)} {emotion}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="prob-track">
                          <div 
                            className="prob-fill"
                            style={{ 
                              width: `${prob * 100}%`,
                              backgroundColor: getEmotionColor(emotion)
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default ResultsDisplay;
