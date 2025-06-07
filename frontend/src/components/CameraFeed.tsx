import { useState, useEffect, useRef } from 'react';

interface CameraFeedProps {
  className?: string;
}

export function CameraFeed({ className = '' }: CameraFeedProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPaused, setIsPaused] = useState(true); // Paused by default
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const fetchFrame = async () => {
    try {
      const response = await fetch('http://localhost:8000/frame');
      
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        // Cleanup previous URL
        if (imageUrl) {
          URL.revokeObjectURL(imageUrl);
        }
        
        setImageUrl(url);
        setError(null);
        setIsLoading(false);
      } else {
        setError('No camera feed available');
        setIsLoading(false);
      }
    } catch (err) {
      setError('Failed to load camera feed');
      setIsLoading(false);
      console.error('Camera feed error:', err);
    }
  };

  const startFeed = () => {
    setIsLoading(true);
    setError(null);
    setIsPaused(false);
    
    // Initial fetch
    fetchFrame();
    
    // Set up interval to refresh feed
    intervalRef.current = setInterval(fetchFrame, 100); // 10 FPS
  };

  const stopFeed = () => {
    setIsPaused(true);
    setIsLoading(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      setImageUrl(null);
    }
  };

  const toggleFeed = () => {
    if (isPaused) {
      startFeed();
    } else {
      stopFeed();
    }
  };
  
  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, []);

  // Paused state (default)
  if (isPaused) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-800/50 rounded-xl flex items-center justify-center relative ${className}`}>
        <div className="text-gray-300 text-center p-8">
          <div className="w-20 h-20 bg-gradient-to-br from-gray-700 to-gray-800 rounded-2xl flex items-center justify-center text-3xl mx-auto mb-6 shadow-lg">
            📷
          </div>
          <h3 className="text-lg font-semibold text-gray-200 mb-2">Camera Feed</h3>
          <p className="text-gray-400 mb-6">Camera feed is paused</p>
          <button 
            onClick={toggleFeed}
            className="bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white px-6 py-3 rounded-lg text-sm font-medium transition-all duration-200 shadow-lg hover:shadow-green-500/25 flex items-center gap-3 mx-auto"
          >
            <span className="text-lg">▶️</span>
            Start Camera
          </button>
        </div>
      </div>
    );
  }
  
  if (isLoading) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-800/50 rounded-xl flex items-center justify-center relative ${className}`}>
        <div className="text-gray-300 text-center p-8">
          <div className="w-12 h-12 border-3 border-gray-700 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-300 font-medium">Loading camera feed...</p>
        </div>
        <button 
          onClick={toggleFeed}
          className="absolute top-4 right-4 glass-card p-2.5 rounded-lg transition-all duration-200 hover:bg-gray-800/80 text-white"
          title="Stop camera"
        >
          <span className="text-lg">⏸️</span>
        </button>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-red-800/30 rounded-xl flex items-center justify-center relative ${className}`}>
        <div className="text-center p-8">
          <div className="w-16 h-16 bg-gradient-to-br from-red-700 to-red-800 rounded-2xl flex items-center justify-center text-2xl mx-auto mb-6 shadow-lg">
            📷
          </div>
          <h3 className="text-lg font-semibold text-red-300 mb-2">Camera Error</h3>
          <p className="text-red-400 mb-6">{error}</p>
          <button 
            onClick={toggleFeed}
            className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white px-6 py-3 rounded-lg text-sm font-medium transition-all duration-200 shadow-lg hover:shadow-blue-500/25 flex items-center gap-3 mx-auto"
          >
            <span className="text-lg">🔄</span>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-900 rounded-xl overflow-hidden relative ${className}`}>
      {imageUrl && (
        <img
          src={imageUrl}
          alt="Camera feed"
          className="w-full h-full object-cover"
          onError={() => setError('Failed to display camera feed')}
        />
      )}
      <button 
        onClick={toggleFeed}
        className="absolute top-4 right-4 glass-card p-2.5 rounded-lg transition-all duration-200 hover:bg-gray-800/80 text-white shadow-lg"
        title="Pause camera"
      >
        <span className="text-lg">⏸️</span>
      </button>
    </div>
  );
} 