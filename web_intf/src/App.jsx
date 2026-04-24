import { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import ResultsDisplay from './components/ResultsDisplay';
import { checkHealth } from './services/api';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  const [isLiveMode, setIsLiveMode] = useState(false); // New: Live stream mode toggle

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

  const handleResults = (newResults, newImagePreview) => {
    // Clear previous results when new image is being processed
    if (!newResults && !newImagePreview) {
      setResults(null);
      setImagePreview(null);
    } else {
      setResults(newResults);
      setImagePreview(newImagePreview);
    }
  };

  const toggleLiveMode = () => {
    setIsLiveMode(!isLiveMode);
    // Clear results when switching modes
    setResults(null);
    setImagePreview(null);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>🐕 Dog Emotion Recognition System</h1>
        <button 
          className={`live-mode-button ${isLiveMode ? 'active' : ''}`}
          onClick={toggleLiveMode}
        >
          {isLiveMode ? '📹 Exit Live Mode' : '📹 Live-Stream Mode'}
        </button>
      </header>

      {/* Main Content - Unified Upload and Results Area */}
      <main className="app-main">
        <div className="unified-container">
          {/* Hide ImageUploader in live mode */}
          {!isLiveMode && (
            <ImageUploader onResults={handleResults} />
          )}
          
          {/* Show ResultsDisplay in both modes, but behavior differs */}
          {(results && imagePreview && !isLiveMode) || isLiveMode ? (
            <ResultsDisplay 
              results={results}
              imagePreview={imagePreview}
              isLiveMode={isLiveMode}
            />
          ) : null}
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
