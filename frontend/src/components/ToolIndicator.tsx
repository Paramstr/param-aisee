import { ToolState } from '@/lib/useSocket';
import { useEffect } from 'react';

interface ToolIndicatorProps {
  toolState: ToolState;
  className?: string;
}

export function ToolIndicator({ toolState, className = '' }: ToolIndicatorProps) {
  
  // Debug logging
  useEffect(() => {
    console.log('🔧 ToolIndicator render:', toolState);
    if (toolState.tool_name === 'get_video') {
      console.log('🎥 Video tool state details:', {
        active: toolState.active,
        action: toolState.action,
        duration: toolState.duration,
        message: toolState.message,
        shouldShowProgressBar: toolState.active && toolState.tool_name === 'get_video' && toolState.action === 'recording' && toolState.duration
      });
    }
  }, [toolState]);

  // Force re-render logging for debugging
  console.log('🔧 ToolIndicator force render check:', {
    active: toolState.active,
    tool_name: toolState.tool_name,
    action: toolState.action,
    timestamp: Date.now()
  });
  
  const getToolIcon = (toolName: string | null) => {
    switch (toolName) {
      case 'get_photo':
        return '📸';
      case 'get_video':
        return '🎥';
      default:
        return '🔧';
    }
  };

  const getStatusColor = (action: string | null) => {
    switch (action) {
      case 'starting':
        return 'bg-blue-500 shadow-blue-500/50';
      case 'capturing':
        return 'bg-green-500 shadow-green-500/50';
      case 'recording':
        return 'bg-red-500 shadow-red-500/50 animate-pulse';
      case 'complete':
        return 'bg-purple-500 shadow-purple-500/50';
      case 'error':
        return 'bg-red-600 shadow-red-600/50';
      default:
        return 'bg-gray-500 shadow-gray-500/50';
    }
  };

  const getStatusText = (action: string | null) => {
    switch (action) {
      case 'starting':
        return 'Preparing...';
      case 'capturing':
        return 'Capturing';
      case 'recording':
        return '🔴 RECORDING';
      case 'complete':
        return 'Complete';
      case 'error':
        return 'Error';
      default:
        return 'Processing';
    }
  };

  // Don't show indicator if no active tool or recent completion
  if (!toolState.active && !toolState.tool_name) {
    console.log('🔧 ToolIndicator: Not showing - no active tool and no tool_name');
    return null;
  }
  
  // Show indicator for active tools or recent completions
  const shouldShow = toolState.active || toolState.tool_name;
  if (!shouldShow) {
    console.log('🔧 ToolIndicator: Not showing - shouldShow is false');
    return null;
  }

  return (
    <div className={`glass-card rounded-xl px-4 py-3 text-white shadow-xl fade-in ${
      toolState.action === 'recording' ? 'ring-2 ring-red-500 bg-red-900/20' : ''
    } ${className}`}>
      <div className="flex items-center gap-3">
        {/* Tool icon */}
        <div className={`w-8 h-8 bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg flex items-center justify-center shadow-lg ${
          toolState.action === 'recording' ? 'animate-pulse' : ''
        }`}>
          <span className="text-sm">{getToolIcon(toolState.tool_name)}</span>
        </div>
        
        {/* Status content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {/* Animated pulse indicator */}
            <div className={`w-2 h-2 rounded-full shadow-lg ${getStatusColor(toolState.action)} ${toolState.active ? 'animate-pulse' : ''}`} />
            <span className={`text-sm font-medium ${
              toolState.action === 'recording' ? 'text-red-200 font-bold' : 'text-gray-100'
            }`}>
              {toolState.message || getStatusText(toolState.action)}
            </span>
          </div>
          
          {/* Tool name subtitle */}
          <div className="text-xs text-gray-300 font-medium uppercase tracking-wide">
            {toolState.tool_name?.replace('get_', '') || 'Tool'}
          </div>
        </div>
        
        {/* Duration indicator for video */}
        {toolState.tool_name === 'get_video' && toolState.duration && (
          <div className={`text-xs text-gray-300 font-mono bg-gray-800/50 px-2 py-1 rounded-md border border-gray-700/50 ${
            toolState.action === 'recording' ? 'animate-pulse text-red-300 border-red-500/50' : ''
          }`}>
            {toolState.duration}s
          </div>
        )}
      </div>
      
      {/* Progress bar for video recording */}
      {toolState.active && toolState.tool_name === 'get_video' && toolState.action === 'recording' && toolState.duration && (
        <div className="mt-3 w-full bg-gray-800/50 rounded-full h-2 border border-red-500/30 shadow-lg">
          <div 
            key={`progress-${toolState.action}-${toolState.duration}`}
            className="bg-gradient-to-r from-red-500 to-red-600 h-full rounded-full shadow-lg shadow-red-500/50 animate-pulse"
            style={{
              animation: `progressBar ${toolState.duration}s linear forwards`,
              animationDelay: '0s'
            }}
          />
        </div>
      )}
      
      <style jsx>{`
        @keyframes progressBar {
          from { 
            width: 0%; 
            opacity: 1;
          }
          to { 
            width: 100%; 
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
} 