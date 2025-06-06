import React from 'react';

interface MediaModalProps {
  isOpen: boolean;
  type: 'image' | 'video';
  url: string;
  duration?: number;
  onClose: () => void;
}

export function MediaModal({ isOpen, type, url, duration, onClose }: MediaModalProps) {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div 
        className="bg-gray-900 p-4 rounded-lg shadow-xl relative max-w-[95vw] max-h-[95vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-3 right-3 text-white bg-gray-700 hover:bg-gray-600 rounded-full p-1 text-2xl leading-none z-10 w-8 h-8 flex items-center justify-center"
          aria-label={`Close ${type} viewer`}
        >
          &times;
        </button>
        
        {type === 'image' ? (
          <img 
            src={url} 
            alt="Enlarged view" 
            className="max-w-full max-h-full object-contain rounded" 
          />
        ) : (
          <div className="flex flex-col items-center">
            <video
              controls
              autoPlay
              className="max-w-full max-h-[80vh] rounded"
              style={{ maxWidth: '90vw' }}
            >
              <source src={url} type="video/mp4" />
              Your browser does not support video playback.
            </video>
            {duration && (
              <div className="text-white text-center mt-2 text-sm">
                Duration: {duration}s
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
} 