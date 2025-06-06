import { useState, useRef, useEffect } from 'react';
import { Event } from '@/lib/useSocket';

interface RawTranscript {
  id: string;
  content: string;
  timestamp: Date;
  hasWakeWord: boolean;
  isProcessed: boolean;
  isInContext: boolean;
}

interface DebugStreamProps {
  lastEvent: Event | null;
  className?: string;
}

export function DebugStream({ lastEvent, className = '' }: DebugStreamProps) {
  const [rawTranscripts, setRawTranscripts] = useState<RawTranscript[]>([]);
  const [showDebugPanel, setShowDebugPanel] = useState(true);
  
  const isTranscribing = useRef(false);
  const isInContextMode = useRef(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  
  // Check if transcript contains wake words
  const containsWakeWord = (transcript: string): boolean => {
    const wakeWords = ["osmo", "hey osmo", "hey"];
    const lower = transcript.toLowerCase();
    return wakeWords.some(word => lower.includes(word));
  };
  
  useEffect(() => {
    if (!lastEvent) return;
    
    const eventType = lastEvent.type;
    const eventAction = lastEvent.action;
    
    switch (eventType) {
      case 'audio_event':
        switch (eventAction) {
          case 'transcription_start':
            isTranscribing.current = true;
            break;
            
          case 'transcription_end':
            isTranscribing.current = false;
            break;
            
          case 'raw_transcript':
            const rawTranscript = lastEvent.data?.transcript;
            if (rawTranscript && rawTranscript.trim()) {
              const hasWakeWord = containsWakeWord(rawTranscript);
              const newRawTranscript: RawTranscript = {
                id: Date.now().toString(),
                content: rawTranscript,
                timestamp: new Date(),
                hasWakeWord,
                isProcessed: false,
                isInContext: isInContextMode.current && !hasWakeWord
              };
              setRawTranscripts(prev => {
                const updated = [...prev, newRawTranscript];
                // Keep only last 30 transcripts
                return updated.slice(-30);
              });
            }
            break;
            
          case 'wake_word_detected':
            // Start context accumulation mode
            isInContextMode.current = true;
            
            // Mark the most recent transcript as having triggered wake word detection
            setRawTranscripts(prev => 
              prev.map((transcript, index) => 
                index === prev.length - 1 ? { ...transcript, isProcessed: true } : transcript
              )
            );
            break;
            
          case 'context_ready':
            // Context accumulation is now complete
            isInContextMode.current = false;
            isTranscribing.current = false;
            break;
        }
        break;
    }
  }, [lastEvent]);
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  
  if (!showDebugPanel) {
    return (
      <div className={`bg-gray-900 rounded-lg p-3 sm:p-4 h-full flex flex-col justify-center ${className}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center min-w-0 flex-1">
            <div className="text-lg sm:text-xl mr-2 flex-shrink-0">🔍</div>
            <h2 className="text-sm sm:text-lg font-semibold text-white truncate">Debug Stream</h2>
          </div>
          <button
            onClick={() => setShowDebugPanel(true)}
            className="flex-shrink-0 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-2 py-1 rounded transition-colors"
          >
            Show
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className={`bg-gray-900 rounded-lg flex flex-col h-full overflow-hidden ${className}`}>
      <div className="flex items-center justify-between p-3 sm:p-4 pb-2 border-b border-gray-700 flex-shrink-0">
        <div className="flex items-center min-w-0 flex-1">
          <div className="text-lg sm:text-xl mr-2 flex-shrink-0">🔍</div>
          <h2 className="text-sm sm:text-lg font-semibold text-white truncate">Debug Stream</h2>
          {isTranscribing.current && (
            <div className="ml-2 flex items-center">
              <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-1"></div>
              <span className="text-xs text-blue-400 hidden sm:inline">Transcribing...</span>
            </div>
          )}
          {isInContextMode.current && (
            <div className="ml-2 flex items-center">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse mr-1"></div>
              <span className="text-xs text-yellow-400 hidden sm:inline">Context Mode</span>
            </div>
          )}
        </div>
        <button
          onClick={() => setShowDebugPanel(false)}
          className="flex-shrink-0 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-2 py-1 rounded transition-colors"
        >
          Hide
        </button>
      </div>
      
      <div 
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto p-3 sm:p-4 pt-3 space-y-1 text-xs min-h-0 max-h-full"
      >
        {rawTranscripts.length === 0 && (
          <div className="text-gray-500 text-center py-2">
            Listening for speech...
          </div>
        )}
        
        {rawTranscripts.map((transcript) => (
          <div
            key={transcript.id}
            className={`p-2 rounded border-l-2 ${
              transcript.hasWakeWord
                ? transcript.isProcessed
                  ? 'bg-green-900/30 border-green-400 text-green-200'
                  : 'bg-yellow-900/30 border-yellow-400 text-yellow-200'
                : transcript.isInContext
                  ? 'bg-blue-900/30 border-blue-400 text-blue-200'
                  : 'bg-gray-700/50 border-gray-600 text-gray-300'
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="flex-1 break-words min-w-0">{transcript.content}</span>
              <div className="flex items-center space-x-1 flex-shrink-0">
                {transcript.hasWakeWord && (
                  <span className={`px-1 py-0.5 rounded text-xs font-medium ${
                    transcript.isProcessed 
                      ? 'bg-green-600 text-green-100' 
                      : 'bg-yellow-600 text-yellow-100'
                  }`}>
                    {transcript.isProcessed ? '→ AI' : 'WAKE'}
                  </span>
                )}
                {transcript.isInContext && (
                  <span className="px-1 py-0.5 rounded text-xs font-medium bg-blue-600 text-blue-100">
                    CTX
                  </span>
                )}
                <span className="text-gray-400 text-xs">
                  {formatTime(transcript.timestamp)}
                </span>
              </div>
            </div>
          </div>
        ))}

      </div>
    </div>
  );
} 