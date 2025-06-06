import { useState, useEffect, useRef } from 'react';
import { Event } from '@/lib/useSocket';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp: Date;
  isComplete: boolean;
  imageBase64?: string;
  videoBase64?: string;
  duration?: number;
  toolName?: string;
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
  const [currentResponseImageBase64, setCurrentResponseImageBase64] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState('');
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState('');
  const [showDebugPanel, setShowDebugPanel] = useState(true);
  const [isImagePopupOpen, setIsImagePopupOpen] = useState(false);
  const [popupImageUrl, setPopupImageUrl] = useState<string | null>(null);
  const [isVideoPopupOpen, setIsVideoPopupOpen] = useState(false);
  const [popupVideoData, setPopupVideoData] = useState<{url: string; duration?: number} | null>(null);
  
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
            const transcript = lastEvent.data?.transcript;
            let shouldAddNewUserMessage = true;

            if (transcript) {
              const lastClientMessage = messages[messages.length - 1];
              // If the last message in UI is a user message and has the exact same content,
              // assume this event is the backend confirmation of a manually sent message
              // that was already added optimistically. Avoid adding it again.
              if (
                lastClientMessage &&
                lastClientMessage.type === 'user' &&
                lastClientMessage.content === transcript
              ) {
                shouldAddNewUserMessage = false;
              }

              if (shouldAddNewUserMessage) {
                const userMessage: Message = {
                  id: Date.now().toString(),
                  type: 'user',
                  content: transcript,
                  timestamp: new Date(),
                  isComplete: true
                };
                setMessages(prev => [...prev, userMessage]);
              }
            }
            
            // Context accumulation (whether voice or triggered by manual send) is now complete.
            // Reset related states.
            isInContextMode.current = false;
            currentContext.current = '';
            // Setting isTranscribing to false here. This is generally correct as context_ready 
            // implies transcription leading to it is done. If text input heavily overlaps active 
            // voice transcription, this might need a more nuanced handling, but for typical 
            // text sends, this is fine.
            isTranscribing.current = false; 
            break;
        }
        break;
        
      case 'llm_event':
        switch (eventAction) {
          case 'response_start':
            setCurrentResponse('');
            isProcessing.current = true;
            setCurrentResponseImageBase64(null);
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
                isComplete: true,
                imageBase64: currentResponseImageBase64 !== null ? currentResponseImageBase64 : undefined
              };
              setMessages(prev => [...prev, assistantMessage]);
              setCurrentResponse('');
              isProcessing.current = false;
              setCurrentResponseImageBase64(null);
            }
            break;
        }
        break;
        
      case 'tool_event':
        switch (eventAction) {
          case 'video_start':
            const duration = lastEvent.data?.duration || 3;
            const toolMessage: Message = {
              id: 'video_' + Date.now().toString(),
              type: 'assistant',
              content: `🎥 Recording ${duration}-second video to analyze...`,
              timestamp: new Date(),
              isComplete: true
            };
            setMessages(prev => [...prev, toolMessage]);
            break;
            
          case 'photo_start':
            const photoMessage: Message = {
              id: 'photo_' + Date.now().toString() + '_' + Math.random().toString(36).substr(2, 9),
              type: 'assistant',
              content: `📸 Capturing photo for analysis...`,
              timestamp: new Date(),
              isComplete: true
            };
            console.log('📸 Photo start - creating message:', photoMessage);
            setMessages(prev => [...prev, photoMessage]);
            break;
            
          case 'photo_complete':
            const capturedPhotoData = lastEvent.data;
            console.log('📸 Photo complete event received:', capturedPhotoData);
            
            if (capturedPhotoData?.success) {
              console.log('📸 Photo success, updating messages');
              
              // Update the photo_start message with success indicator
              setMessages(prev => {
                const updated = prev.map(msg => {
                  if (msg.id.startsWith('photo_') && msg.content.includes('Capturing photo for analysis')) {
                    console.log('📸 Updating message:', msg.id, msg.content);
                    return { ...msg, content: `✅ Photo captured successfully` };
                  }
                  return msg;
                });
                console.log('📸 Messages after update:', updated);
                return updated;
              });
              
              // Add the captured photo as a tool message
              if (capturedPhotoData.photo_base64) {
                console.log('📸 Adding photo tool message');
                const photoToolMessage: Message = {
                  id: Date.now().toString(),
                  type: 'tool',
                  content: 'Photo captured',
                  timestamp: new Date(),
                  isComplete: true,
                  imageBase64: capturedPhotoData.photo_base64,
                  toolName: 'get_photo'
                };
                setMessages(prev => [...prev, photoToolMessage]);
              }
            } else {
              console.log('📸 Photo failed, updating with error');
              // Update with error indicator
              setMessages(prev => prev.map(msg => 
                msg.id.startsWith('photo_') && msg.content.includes('Capturing photo for analysis')
                  ? { ...msg, content: `❌ Photo capture failed` }
                  : msg
              ));
            }
            break;
            
          case 'video_complete':
            const videoData = lastEvent.data;
            if (videoData?.success) {
              // Update the video_start message with success indicator
              setMessages(prev => prev.map(msg => 
                msg.id.startsWith('video_') && msg.content.includes('Recording') && msg.content.includes('video to analyze')
                  ? { ...msg, content: `✅ Video recorded successfully (${videoData.duration}s)` }
                  : msg
              ));
              
              // Add the captured video as a tool message
              if (videoData.video_base64) {
                const videoMessage: Message = {
                  id: Date.now().toString(),
                  type: 'tool',
                  content: `Video captured (${videoData.duration}s, ${videoData.frames_recorded} frames)`,
                  timestamp: new Date(),
                  isComplete: true,
                  videoBase64: videoData.video_base64,
                  duration: videoData.duration,
                  toolName: 'get_video'
                };
                setMessages(prev => [...prev, videoMessage]);
              }
            } else {
              // Update with error indicator
              setMessages(prev => prev.map(msg => 
                msg.id.startsWith('video_') && msg.content.includes('Recording') && msg.content.includes('video to analyze')
                  ? { ...msg, content: `❌ Video recording failed` }
                  : msg
              ));
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
  
  // Image popup handlers
  const openImagePopup = (imageUrl: string) => {
    setPopupImageUrl(imageUrl);
    setIsImagePopupOpen(true);
  };

  const closeImagePopup = () => {
    setIsImagePopupOpen(false);
    setPopupImageUrl(null);
  };
  
  // Video popup handlers
  const openVideoPopup = (videoUrl: string, duration?: number) => {
    setPopupVideoData({ url: videoUrl, duration });
    setIsVideoPopupOpen(true);
  };

  const closeVideoPopup = () => {
    setIsVideoPopupOpen(false);
    setPopupVideoData(null);
  };
  
  return (
    <div className={`bg-gray-900 rounded-lg p-4 flex flex-col max-h-full ${className}`}>
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
          
          <div className="h-32 overflow-y-auto space-y-1 text-xs">
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
      
      {/* Main Conversation - Scrollable with fixed height */}
      <div 
        ref={conversationRef}
        className="flex-1 overflow-y-auto space-y-3 min-h-0 pr-1 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800"
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
                  : message.type === 'tool'
                  ? 'bg-purple-600 text-white border-l-4 border-purple-400'
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
                  <div className="mb-1">
                    {/* Tool message with media */}
                    {message.type === 'tool' && (
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-sm font-medium">
                          <span>{message.toolName === 'get_video' ? '🎥' : '📸'}</span>
                          <span>{message.content}</span>
                        </div>
                        
                        {/* Display captured video */}
                        {message.videoBase64 && (
                          <div className="mt-2">
                            <video
                              controls
                              className="rounded-lg max-w-full max-h-[200px] bg-black cursor-pointer"
                              preload="metadata"
                              onClick={() => openVideoPopup(`data:video/mp4;base64,${message.videoBase64}`, message.duration)}
                            >
                              <source 
                                src={`data:video/mp4;base64,${message.videoBase64}`} 
                                type="video/mp4" 
                              />
                              Your browser does not support video playback.
                            </video>
                            <div className="text-xs text-purple-200 mt-1">
                              Duration: {message.duration}s • Click to enlarge
                            </div>
                          </div>
                        )}
                        
                        {/* Display captured photo */}
                        {message.imageBase64 && (
                          <div className="mt-2">
                            <img
                              src={`data:image/jpeg;base64,${message.imageBase64}`}
                              alt="Captured photo"
                              className="rounded-lg max-w-[200px] max-h-[200px] cursor-pointer hover:opacity-80 transition-opacity"
                              onClick={() => openImagePopup(`data:image/jpeg;base64,${message.imageBase64}`)}
                            />
                            <div className="text-xs text-purple-200 mt-1">
                              Click to enlarge
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Regular user/assistant messages */}
                    {message.type !== 'tool' && (
                      <>
                        {message.type === 'user' && message.imageBase64 ? (
                          <div className="flex items-start space-x-2">
                            <img
                              src={`data:image/jpeg;base64,${message.imageBase64}`}
                              alt="User context"
                              className="rounded-md max-w-[80px] max-h-[80px] cursor-pointer hover:opacity-80 transition-opacity"
                              onClick={() => openImagePopup(`data:image/jpeg;base64,${message.imageBase64}`)}
                            />
                            <div className="text-sm flex-1">{message.content}</div>
                          </div>
                        ) : (
                          <div className="text-sm">
                            {message.content}
                          </div>
                        )}

                        {message.type === 'assistant' && message.imageBase64 && (
                          <img
                            src={`data:image/jpeg;base64,${message.imageBase64}`}
                            alt="Assistant context"
                            className="mt-2 rounded-md max-w-[150px] max-h-[150px] cursor-pointer hover:opacity-80 transition-opacity"
                            onClick={() => openImagePopup(`data:image/jpeg;base64,${message.imageBase64}`)}
                          />
                        )}
                      </>
                    )}
                  </div>

                  <div className="flex items-center justify-between">
                    <div
                      className={`text-xs opacity-70 ${
                        message.type === 'user' ? 'text-blue-200' : 
                        message.type === 'tool' ? 'text-purple-200' :
                        'text-gray-400'
                      }`}
                    >
                      {formatTime(message.timestamp)}
                    </div>
                    {message.type === 'user' && (
                      <button
                        onClick={() => startEditingMessage(message.id, message.content)}
                        className="text-xs opacity-0 group-hover:opacity-100 transition-opacity hover:bg-gray-600 px-1 py-0.5 rounded"
                      >
                        ✏️
                      </button>
                    )}
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
                {currentResponseImageBase64 && (
                  <img
                    src={`data:image/jpeg;base64,${currentResponseImageBase64}`}
                    alt="Current context"
                    className="mt-2 rounded-md max-w-[150px] max-h-[150px]"
                  />
                )}
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

      {/* Image Popup Modal */}
      {isImagePopupOpen && popupImageUrl && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={closeImagePopup}
        >
          <div 
            className="bg-gray-900 p-4 rounded-lg shadow-xl relative max-w-full max-h-full"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeImagePopup}
              className="absolute top-3 right-3 text-white bg-gray-700 hover:bg-gray-600 rounded-full p-1 text-2xl leading-none z-10"
              aria-label="Close image viewer"
            >
              &times;
            </button>
            <img 
              src={popupImageUrl} 
              alt="Enlarged view" 
              className="max-w-[90vw] max-h-[90vh] object-contain rounded" 
            />
          </div>
        </div>
      )}

      {/* Video Popup Modal */}
      {isVideoPopupOpen && popupVideoData && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={closeVideoPopup}
        >
          <div 
            className="bg-gray-900 p-4 rounded-lg shadow-xl relative max-w-full max-h-full"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeVideoPopup}
              className="absolute top-3 right-3 text-white bg-gray-700 hover:bg-gray-600 rounded-full p-1 text-2xl leading-none z-10"
              aria-label="Close video viewer"
            >
              &times;
            </button>
            <video
              controls
              autoPlay
              className="max-w-[90vw] max-h-[90vh] rounded"
            >
              <source src={popupVideoData.url} type="video/mp4" />
              Your browser does not support video playback.
            </video>
            {popupVideoData.duration && (
              <div className="text-white text-center mt-2 text-sm">
                Duration: {popupVideoData.duration}s
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
} 