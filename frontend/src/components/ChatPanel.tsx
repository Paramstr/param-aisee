import { useState, useEffect, useRef } from 'react';
import { Event } from '@/lib/useSocket';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isComplete: boolean;
}

interface RawTranscript {
  id: string;
  content: string;
  timestamp: Date;
  hasWakeWord: boolean;
  isProcessed: boolean;
  isInContext: boolean;
}

interface ChatPanelProps {
  lastEvent: Event | null;
  className?: string;
}

export function ChatPanel({ lastEvent, className = '' }: ChatPanelProps) {
  // Only use useState for values that need to trigger re-renders
  const [messages, setMessages] = useState<Message[]>([]);
  const [rawTranscripts, setRawTranscripts] = useState<RawTranscript[]>([]);
  const [currentResponse, setCurrentResponse] = useState<string>('');
  const [inputMessage, setInputMessage] = useState('');
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState('');
  const [showDebugPanel, setShowDebugPanel] = useState(true);
  
  // Use refs for state that doesn't need re-renders or is derived
  const isProcessing = useRef(false);
  const isTranscribing = useRef(false);
  const isInContextMode = useRef(false);
  const currentContext = useRef<string>('');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const debugEndRef = useRef<HTMLDivElement>(null);
  const conversationRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const editInputRef = useRef<HTMLTextAreaElement>(null);
  
  // Derived state for UI status
  const getStatusInfo = () => {
    if (isTranscribing.current) return { color: 'bg-blue-400', text: 'Transcribing speech...' };
    if (isProcessing.current) return { color: 'bg-yellow-400', text: 'Osmo is thinking...' };
    if (isInContextMode.current) return { color: 'bg-orange-400', text: 'Accumulating context (2s silence to process)...' };
    return { color: 'bg-green-400', text: 'Ready to listen' };
  };
  
  // Auto-scroll to bottom with better behavior
  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      if (showDebugPanel) {
        debugEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages, currentResponse, rawTranscripts, showDebugPanel]);
  
  // Focus edit input when editing starts
  useEffect(() => {
    if (editingMessageId && editInputRef.current) {
      editInputRef.current.focus();
    }
  }, [editingMessageId]);
  
  // Check if transcript contains wake words
  const containsWakeWord = (transcript: string): boolean => {
    const wakeWords = ["osmo", "hey osmo", "testing"];
    const lower = transcript.toLowerCase();
    return wakeWords.some(word => lower.includes(word));
  };
  
  // Handle manual message send
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date(),
      isComplete: true
    };
    
    setMessages(prev => [...prev, userMessage]);
    const messageToSend = inputMessage.trim();
    setInputMessage('');
    
    try {
      // Send to backend API for LLM processing
      const response = await fetch('http://localhost:8000/conversation/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: messageToSend
        }),
      });
      
      if (!response.ok) {
        // Try to get error details from response
        let errorMessage = `Server error (${response.status})`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          // If can't parse JSON, use status text
          errorMessage = `${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }
      
      const result = await response.json();
      console.log('Message sent successfully:', result);
      
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Determine error message based on error type
      let errorContent = 'Unknown error occurred';
      
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorContent = 'Cannot connect to Osmo Assistant. Please make sure the backend is running on http://localhost:8000';
      } else if (error instanceof Error) {
        errorContent = `Error: ${error.message}`;
      }
      
      // Add error message to conversation
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: errorContent,
        timestamp: new Date(),
        isComplete: true
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };
  
  // Handle message editing
  const startEditingMessage = (messageId: string, content: string) => {
    setEditingMessageId(messageId);
    setEditingContent(content);
  };
  
  const saveEditedMessage = () => {
    if (!editingMessageId || !editingContent.trim()) return;
    
    setMessages(prev => prev.map(msg => 
      msg.id === editingMessageId 
        ? { ...msg, content: editingContent.trim() }
        : msg
    ));
    
    setEditingMessageId(null);
    setEditingContent('');
  };
  
  const cancelEditing = () => {
    setEditingMessageId(null);
    setEditingContent('');
  };
  
  // Handle key presses
  const handleInputKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const handleEditKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveEditedMessage();
    } else if (e.key === 'Escape') {
      cancelEditing();
    }
  };
  
  useEffect(() => {
    if (!lastEvent) return;
    
    // Handle new consolidated event system: type:action
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
              
              // Update current context if in accumulation mode
              if (isInContextMode.current) {
                currentContext.current = currentContext.current 
                  ? `${currentContext.current} ${rawTranscript}` 
                  : rawTranscript;
              }
            }
            break;
            
          case 'wake_word_detected':
            // Start context accumulation mode
            isInContextMode.current = true;
            currentContext.current = lastEvent.data?.transcript || '';
            
            // Mark the most recent transcript as having triggered wake word detection
            setRawTranscripts(prev => 
              prev.map((transcript, index) => 
                index === prev.length - 1 ? { ...transcript, isProcessed: true } : transcript
              )
            );
            break;
            
          case 'context_ready':
            // Context accumulation complete, transcript ready for LLM
            const transcript = lastEvent.data?.transcript;
            if (transcript) {
              const userMessage: Message = {
                id: Date.now().toString(),
                type: 'user',
                content: transcript,
                timestamp: new Date(),
                isComplete: true
              };
              setMessages(prev => [...prev, userMessage]);
            }
            
            // Reset context mode and display
            isInContextMode.current = false;
            currentContext.current = '';
            isTranscribing.current = false;
            break;
        }
        break;
        
      case 'llm_event':
        switch (eventAction) {
          case 'response_start':
            setCurrentResponse('');
            isProcessing.current = true;
            break;
            
          case 'response_chunk':
            const chunk = lastEvent.data?.chunk;
            if (chunk) {
              setCurrentResponse(prev => prev + chunk);
            }
            break;
            
          case 'response_end':
            const fullResponse = lastEvent.data?.full_response;
            if (fullResponse) {
              const assistantMessage: Message = {
                id: Date.now().toString(),
                type: 'assistant',
                content: fullResponse,
                timestamp: new Date(),
                isComplete: true
              };
              setMessages(prev => [...prev, assistantMessage]);
              setCurrentResponse('');
              isProcessing.current = false;
            }
            break;
        }
        break;
    }
  }, [lastEvent]);
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  
  const statusInfo = getStatusInfo();
  
  return (
    <div className={`bg-gray-900 rounded-lg p-4 flex flex-col h-full ${className}`}>
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-gray-700 flex-shrink-0">
        <div className="flex items-center">
          <div className="text-xl mr-2">💬</div>
          <h2 className="text-lg font-semibold text-white">Conversation</h2>
          {isInContextMode.current && (
            <div className="ml-3 flex items-center text-yellow-400">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse mr-1"></div>
              <span className="text-xs">Accumulating context...</span>
            </div>
          )}
        </div>
        <button
          onClick={() => setShowDebugPanel(!showDebugPanel)}
          className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-2 py-1 rounded transition-colors"
        >
          {showDebugPanel ? 'Hide' : 'Show'} Debug
        </button>
      </div>
      
      {/* Debug Panel for Raw Transcriptions */}
      {showDebugPanel && (
        <div className="mb-4 border border-gray-700 rounded-lg p-3 bg-gray-800 flex-shrink-0">
          <div className="flex items-center mb-2">
            <div className="text-sm mr-2">🔍</div>
            <h3 className="text-sm font-medium text-gray-300">Whisper Debug Stream</h3>
            {isTranscribing.current && (
              <div className="ml-2 flex items-center">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-1"></div>
                <span className="text-xs text-blue-400">Transcribing...</span>
              </div>
            )}
            {isInContextMode.current && (
              <div className="ml-2 flex items-center">
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse mr-1"></div>
                <span className="text-xs text-yellow-400">Context Mode</span>
              </div>
            )}
          </div>
          
          <div className="h-40 overflow-y-auto space-y-1 text-xs">
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
                <div className="flex items-center justify-between">
                  <span className="flex-1 mr-2">{transcript.content}</span>
                  <div className="flex items-center space-x-1">
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
            <div ref={debugEndRef} />
          </div>
        </div>
      )}
      
      {/* Context Accumulation Display in Main Conversation */}
      {isInContextMode.current && currentContext.current && (
        <div className="mb-4 p-3 bg-yellow-900/20 border border-yellow-400/30 rounded-lg flex-shrink-0">
          <div className="flex items-center mb-2">
            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse mr-2"></div>
            <span className="text-sm text-yellow-300 font-medium">Building Context</span>
          </div>
          <div className="text-sm text-yellow-100 bg-yellow-900/30 p-2 rounded">
            {currentContext.current}
          </div>
          <div className="text-xs text-yellow-400 mt-1">
            Continue speaking or wait 2 seconds to send to AI
          </div>
        </div>
      )}
      
      {/* Main Conversation - Now takes up remaining space */}
      <div 
        ref={conversationRef}
        className="flex-1 overflow-y-auto space-y-3 min-h-0 pr-1"
        style={{ scrollBehavior: 'smooth' }}
      >
        {messages.length === 0 && !isProcessing.current && !isInContextMode.current && (
          <div className="text-gray-400 text-center py-8">
            <div className="text-4xl mb-2">🎤</div>
            <p>Say &quot;Osmo&quot; to start a conversation</p>
            <p className="text-sm mt-2 text-gray-500">
              Context will accumulate until 2 seconds of silence
            </p>
            <p className="text-xs mt-3 text-gray-600">
              You can also type messages below ↓
            </p>
          </div>
        )}
        
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-3 py-2 group relative ${
                message.type === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-100'
              }`}
            >
              {editingMessageId === message.id ? (
                <div className="space-y-2">
                  <textarea
                    ref={editInputRef}
                    value={editingContent}
                    onChange={(e) => setEditingContent(e.target.value)}
                    onKeyDown={handleEditKeyPress}
                    className="w-full bg-gray-800 text-white p-2 rounded text-sm resize-none"
                    rows={3}
                  />
                  <div className="flex space-x-2">
                    <button
                      onClick={saveEditedMessage}
                      className="text-xs bg-green-600 hover:bg-green-700 text-white px-2 py-1 rounded"
                    >
                      Save
                    </button>
                    <button
                      onClick={cancelEditing}
                      className="text-xs bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 rounded"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="text-sm mb-1">
                    {message.content}
                  </div>
                  <div className="flex items-center justify-between">
                    <div
                      className={`text-xs opacity-70 ${
                        message.type === 'user' ? 'text-blue-200' : 'text-gray-400'
                      }`}
                    >
                      {formatTime(message.timestamp)}
                    </div>
                    <button
                      onClick={() => startEditingMessage(message.id, message.content)}
                      className="text-xs opacity-0 group-hover:opacity-100 transition-opacity hover:bg-gray-600 px-1 py-0.5 rounded"
                    >
                      ✏️
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        ))}
        
        {/* Current streaming response */}
        {isProcessing.current && (
          <div className="flex justify-start">
            <div className="max-w-[80%] bg-gray-700 text-gray-100 rounded-lg px-3 py-2">
              <div className="text-sm mb-1">
                {currentResponse}
                <span className="animate-pulse">▋</span>
              </div>
              <div className="text-xs text-gray-400 opacity-70">
                {formatTime(new Date())}
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      {/* Manual Input Section */}
      <div className="mt-4 pt-3 border-t border-gray-700 flex-shrink-0">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleInputKeyPress}
            placeholder="Type a message..."
            disabled={isProcessing.current}
            className="flex-1 bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isProcessing.current}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm transition-colors"
          >
            Send
          </button>
        </div>
      </div>
      
      {/* Status indicator - Fixed at bottom */}
      <div className="text-xs text-gray-400 flex items-center mt-3 pt-2 border-t border-gray-700 flex-shrink-0">
        <div className={`w-2 h-2 rounded-full mr-2 ${statusInfo.color} ${statusInfo.color.includes('animate-pulse') || isTranscribing.current || isProcessing.current || isInContextMode.current ? 'animate-pulse' : ''}`} />
        {statusInfo.text}
      </div>
    </div>
  );
} 