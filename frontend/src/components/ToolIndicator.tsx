import { ToolState } from '@/lib/useSocket';

interface ToolIndicatorProps {
  toolState: ToolState;
  className?: string;
}

export function ToolIndicator({ toolState, className = '' }: ToolIndicatorProps) {
  if (!toolState.active && !toolState.message) {
    return null;
  }

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
        return 'bg-blue-500';
      case 'capturing':
        return 'bg-green-500';
      case 'recording':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <div className={`bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3 text-white ${className}`}>
      <div className="flex items-center gap-3">
        {/* Animated pulse indicator */}
        <div className={`w-3 h-3 rounded-full ${getStatusColor(toolState.action)} ${toolState.active ? 'animate-pulse' : ''}`} />
        
        {/* Tool icon and status */}
        <div className="flex items-center gap-2">
          <span className="text-lg">{getToolIcon(toolState.tool_name)}</span>
          <span className="text-sm font-medium">
            {toolState.message || `${toolState.tool_name} ${toolState.action}`}
          </span>
        </div>
        
        {/* Duration indicator for video */}
        {toolState.tool_name === 'get_video' && toolState.duration && (
          <div className="text-xs text-gray-300">
            {toolState.duration}s
          </div>
        )}
      </div>
      
      {/* Progress bar for video recording */}
      {toolState.active && toolState.tool_name === 'get_video' && toolState.duration && (
        <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
          <div 
            className="bg-red-500 h-1 rounded-full transition-all duration-1000 ease-linear"
            style={{
              animation: `progress ${toolState.duration}s linear forwards`
            }}
          />
        </div>
      )}
      
      <style jsx>{`
        @keyframes progress {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </div>
  );
} 