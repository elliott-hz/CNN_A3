import { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import VideoUploader from './components/VideoUploader';
import ResultsDisplay from './components/ResultsDisplay';
import VideoResultsDisplay from './components/VideoResultsDisplay';
import LiveStream from './components/LiveStream';
import { checkHealth } from './services/api';
import './App.css';

// Mode constants
const MODES = {
  IMAGE: 'image',
  VIDEO: 'video',
  LIVE: 'live'
};

function App() {
  const [currentMode, setCurrentMode] = useState(MODES.IMAGE);
  const [results, setResults] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Check API health on mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        await checkHealth();
        setApiStatus('connected');
      } catch (error) {
        setApiStatus('disconnected');
        console.error('API connection failed:', error);
      }
    };
    
    checkApiHealth();
  }, []);

  const handleImageResults = (newResults, newImagePreview) => {
    if (!newResults && !newImagePreview) {
      setResults(null);
      setImagePreview(null);
    } else {
      setResults(newResults);
      setImagePreview(newImagePreview);
    }
  };

  const handleVideoSelect = (videoFile) => {
    setSelectedVideo(videoFile);
    // Clear image results when switching to video mode
    if (videoFile) {
      setResults(null);
      setImagePreview(null);
    }
  };

  const switchMode = (mode) => {
    setCurrentMode(mode);
    // Clear all results when switching modes
    setResults(null);
    setImagePreview(null);
    setSelectedVideo(null);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>🐕 Dog Emotion Recognition System</h1>
        <div className="mode-buttons">
          <button 
            className={`mode-button ${currentMode === MODES.IMAGE ? 'active' : ''}`}
            onClick={() => switchMode(MODES.IMAGE)}
          >
            📷 Upload Image
          </button>
          <button 
            className={`mode-button ${currentMode === MODES.VIDEO ? 'active' : ''}`}
            onClick={() => switchMode(MODES.VIDEO)}
          >
            🎬 Upload Video
          </button>
          <button 
            className={`mode-button ${currentMode === MODES.LIVE ? 'active' : ''}`}
            onClick={() => switchMode(MODES.LIVE)}
          >
            📹 Live Stream
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="unified-container">
          {/* Image Mode */}
          {currentMode === MODES.IMAGE && (
            <>
              <ImageUploader onResults={handleImageResults} />
              {(results && imagePreview) && (
                <ResultsDisplay 
                  results={results}
                  imagePreview={imagePreview}
                  isLiveMode={false}
                />
              )}
            </>
          )}

          {/* Video Mode */}
          {currentMode === MODES.VIDEO && (
            <>
              {!selectedVideo ? (
                <VideoUploader onVideoSelect={handleVideoSelect} />
              ) : (
                <VideoResultsDisplay 
                  videoFile={selectedVideo}
                  isProcessing={false}
                />
              )}
            </>
          )}

          {/* Live Stream Mode */}
          {currentMode === MODES.LIVE && (
            <LiveStream />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Built with FastAPI + React | YOLOv8 + ResNet50</p>
      </footer>
    </div>
  );
}

export default App;
