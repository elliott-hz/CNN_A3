import { useState, useEffect } from 'react';
import ImageUploader from './components/ImageUploader';
import ResultsDisplay from './components/ResultsDisplay';
import { checkHealth } from './services/api';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
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

  const handleResults = (newResults, newImagePreview) => {
    setResults(newResults);
    setImagePreview(newImagePreview);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>🐕 Dog Emotion Recognition System</h1>
        <div className={`status-indicator ${apiStatus}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {apiStatus === 'connected' && '✅ API Connected'}
            {apiStatus === 'disconnected' && '❌ API Disconnected'}
            {apiStatus === 'checking' && '⏳ Connecting...'}
          </span>
        </div>
      </header>

      {/* Main Content - Unified Upload and Results Area */}
      <main className="app-main">
        <div className="unified-container">
          <ImageUploader onResults={handleResults} />
          
          {results && imagePreview && (
            <ResultsDisplay 
              results={results}
              imagePreview={imagePreview}
            />
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
