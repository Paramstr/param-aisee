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
      <div className={`bg-gray-900 rounded-lg flex items-center justify-center relative ${className}`}>
        <div className="text-gray-400 text-center">
          <div className="text-6xl mb-4">📷</div>
          <p className="mb-4">Camera feed paused</p>
          <button 
            onClick={toggleFeed}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2 mx-auto"
          >
            <span>▶️</span>
            Start Camera
          </button>
        </div>
      </div>
    );
  }
  
  if (isLoading) {
    return (
      <div className={`bg-gray-900 rounded-lg flex items-center justify-center relative ${className}`}>
        <div className="text-gray-400 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
          <p>Loading camera feed...</p>
        </div>
        <button 
          onClick={toggleFeed}
          className="absolute top-4 right-4 bg-gray-700 hover:bg-gray-600 text-white p-2 rounded-lg transition-colors"
          title="Stop camera"
        >
          ⏸️
        </button>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className={`bg-gray-900 rounded-lg flex items-center justify-center relative ${className}`}>
        <div className="text-red-400 text-center">
          <div className="text-3xl mb-2">📷</div>
          <p className="mb-4">{error}</p>
          <button 
            onClick={toggleFeed}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2 mx-auto"
          >
            <span>🔄</span>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-900 rounded-lg overflow-hidden relative ${className}`}>
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
        className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 text-white p-2 rounded-lg transition-colors backdrop-blur-sm"
        title="Pause camera"
      >
        ⏸️
      </button>
    </div>
  );
} 