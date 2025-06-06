import { Event, SystemStatus } from '@/lib/useSocket';

interface StatusBarProps {
  isConnected: boolean;
  systemStatus: SystemStatus | null;
  lastEvent: Event | null;
  error: string | null;
  className?: string;
}

export function StatusBar({ 
  isConnected, 
  systemStatus, 
  lastEvent, 
  error, 
  className = '' 
}: StatusBarProps) {
  const getStatusColor = (status: boolean) => {
    return status ? 'text-green-400' : 'text-red-400';
  };
  
  const getStatusIcon = (status: boolean) => {
    return status ? '●' : '○';
  };
  
  const getLastEventDisplay = () => {
    if (!lastEvent) return 'No events';
    
    // Handle new consolidated event system: type:action
    const eventKey = `${lastEvent.type}:${lastEvent.action}`;
    
    const eventDisplayMap: Record<string, string> = {
      // System status events
      'system_status:listening': '🎤 Audio listening started',
      'system_status:camera_active': '📹 Camera activated',
      'system_status:whisper_loading': '🔄 Loading Whisper model',
      'system_status:whisper_ready': '✅ Whisper model ready',
      'system_status:system_ready': '🚀 System ready',
      
      // Audio events
      'audio_event:transcription_start': '🎯 Starting transcription',
      'audio_event:transcription_end': '✅ Transcription complete',
      'audio_event:raw_transcript': '📝 Raw transcript received',
      'audio_event:wake_word_detected': '🎤 Wake word detected',
      'audio_event:context_ready': '📋 Context ready for AI',
      
      // LLM events
      'llm_event:response_start': '🤖 AI thinking',
      'llm_event:response_chunk': '💭 AI responding',
      'llm_event:response_end': '✅ Response complete',
      
      // TTS events
      'tts_event:start': '🔊 Speaking',
      'tts_event:end': '🔇 Speech complete',
      
      // Vision events
      'vision_event:frame_captured': '📷 Frame captured',
      
      // Error events
      'error:camera_init_failed': '❌ Camera initialization failed',
      'error:whisper_failed': '❌ Whisper model failed',
      'error:llm_processing_failed': '❌ AI processing failed',
      'error:tts_failed': '❌ Text-to-speech failed',
      'error:audio_loop_error': '❌ Audio processing error',
    };
    
    return eventDisplayMap[eventKey] || `Event: ${lastEvent.type}:${lastEvent.action}`;
  };
  
  return (
    <div className={`bg-gray-900 rounded-lg flex flex-col overflow-hidden ${className}`}>
      <div className="flex-shrink-0 flex items-center p-3 sm:p-4 pb-2 border-b border-gray-700">
        <div className="text-lg sm:text-xl mr-2">📊</div>
        <h2 className="text-sm sm:text-lg font-semibold text-white">System Status</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-3 sm:p-4 pt-3 space-y-3 sm:space-y-4 min-h-0">
      
        {/* Connection Status */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
          <div className="space-y-2">
            <h3 className="text-xs sm:text-sm font-medium text-gray-300">Connection</h3>
            <div className="flex items-center space-x-2">
              <span className={getStatusColor(isConnected)}>
                {getStatusIcon(isConnected)}
              </span>
              <span className="text-xs sm:text-sm text-gray-300">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            {error && (
              <div className="text-xs text-red-400 break-words">
                {error}
              </div>
            )}
          </div>
          
          <div className="space-y-2">
            <h3 className="text-xs sm:text-sm font-medium text-gray-300">System</h3>
            <div className="flex items-center space-x-2">
              <span className={getStatusColor(systemStatus?.is_running || false)}>
                {getStatusIcon(systemStatus?.is_running || false)}
              </span>
              <span className="text-xs sm:text-sm text-gray-300">
                {systemStatus?.is_running ? 'Running' : 'Stopped'}
              </span>
            </div>
          </div>
        </div>
      
        {/* Component Status */}
        {systemStatus && (
          <div className="space-y-2">
            <h3 className="text-xs sm:text-sm font-medium text-gray-300">Components</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              <div className="flex items-center space-x-2">
                <span className={getStatusColor(systemStatus.audio_listening)}>
                  {getStatusIcon(systemStatus.audio_listening)}
                </span>
                <span className="text-gray-400 truncate">Audio Listening</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className={getStatusColor(systemStatus.vision_capturing)}>
                  {getStatusIcon(systemStatus.vision_capturing)}
                </span>
                <span className="text-gray-400 truncate">Camera Capture</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className={getStatusColor(systemStatus.llm_processing)}>
                  {getStatusIcon(systemStatus.llm_processing)}
                </span>
                <span className="text-gray-400 truncate">AI Processing</span>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className={getStatusColor(systemStatus.whisper_loaded)}>
                  {getStatusIcon(systemStatus.whisper_loaded)}
                </span>
                <span className="text-gray-400 truncate">Whisper Model</span>
              </div>
            </div>
          </div>
        )}
        
        {/* Last Event */}
        <div className="space-y-2">
          <h3 className="text-xs sm:text-sm font-medium text-gray-300">Latest Activity</h3>
          <div className="text-xs text-gray-400 break-words">
            {getLastEventDisplay()}
          </div>
          {lastEvent && (
            <div className="text-xs text-gray-500">
              {new Date(lastEvent.timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 