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
        return 'bg-blue-500 shadow-blue-500/50';
      case 'capturing':
        return 'bg-green-500 shadow-green-500/50';
      case 'recording':
        return 'bg-red-500 shadow-red-500/50';
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
        return 'Recording';
      default:
        return 'Processing';
    }
  };

  return (
    <div className={`glass-card rounded-xl px-4 py-3 text-white shadow-xl fade-in ${className}`}>
      <div className="flex items-center gap-3">
        {/* Tool icon */}
        <div className="w-8 h-8 bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg flex items-center justify-center shadow-lg">
          <span className="text-sm">{getToolIcon(toolState.tool_name)}</span>
        </div>
        
        {/* Status content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            {/* Animated pulse indicator */}
            <div className={`w-2 h-2 rounded-full shadow-lg ${getStatusColor(toolState.action)} ${toolState.active ? 'animate-pulse' : ''}`} />
            <span className="text-sm font-medium text-gray-100">
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
          <div className="text-xs text-gray-300 font-mono bg-gray-800/50 px-2 py-1 rounded-md border border-gray-700/50">
            {toolState.duration}s
          </div>
        )}
      </div>
      
      {/* Progress bar for video recording */}
      {toolState.active && toolState.tool_name === 'get_video' && toolState.duration && (
        <div className="mt-3 w-full bg-gray-800/50 rounded-full h-1.5 border border-gray-700/30">
          <div 
            className="bg-gradient-to-r from-red-500 to-red-600 h-full rounded-full shadow-lg shadow-red-500/30 transition-all duration-1000 ease-linear"
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