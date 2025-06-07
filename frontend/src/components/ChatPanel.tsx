import { useState, useEffect, useRef } from 'react';
import { Event } from '@/lib/useSocket';
import { ChatMessage } from './ChatMessage';
import { MediaModal } from './MediaModal';

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

interface ChatPanelProps {
  lastEvent: Event | null;
  className?: string;
}

export function ChatPanel({ lastEvent, className = '' }: ChatPanelProps) {
  // Core state
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentResponse, setCurrentResponse] = useState<string>('');
  const [currentResponseImageBase64, setCurrentResponseImageBase64] = useState<string | null>(null);
  const [inputMessage, setInputMessage] = useState('');
  
  // Edit state
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingContent, setEditingContent] = useState('');
  
  // Modal state
  const [modalState, setModalState] = useState<{
    isOpen: boolean;
    type: 'image' | 'video';
    url: string;
    duration?: number;
  }>({
    isOpen: false,
    type: 'image',
    url: '',
  });
  
  // Processing state (using refs to avoid re-renders)
  const isProcessing = useRef(false);
  const isTranscribing = useRef(false);
  const isInContextMode = useRef(false);
  const currentContext = useRef<string>('');
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    // Use requestAnimationFrame to ensure DOM has updated
    requestAnimationFrame(() => {
      setTimeout(() => {
        if (messagesContainerRef.current) {
          // Scroll to bottom using scrollTop
          messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
        }
      }, 100);
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages.length, currentResponse]);

  // Status info for UI
  const getStatusInfo = () => {
    if (isTranscribing.current) return { color: 'bg-blue-400', text: 'Transcribing speech...' };
    if (isProcessing.current) return { color: 'bg-yellow-400', text: 'Osmo is thinking...' };
    if (isInContextMode.current) return { color: 'bg-orange-400', text: 'Accumulating context (2s silence to process)...' };
    return { color: 'bg-green-400', text: 'Ready to listen' };
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
    
    const messageToSend = inputMessage.trim();
    setInputMessage('');
    setMessages(prev => [...prev, userMessage]);
    
    try {
      const response = await fetch('http://localhost:8000/conversation/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageToSend }),
      });
      
      if (!response.ok) {
        let errorMessage = `Server error (${response.status})`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch {
          errorMessage = `${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      let errorContent = 'Unknown error occurred';
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorContent = 'Cannot connect to Osmo Assistant. Please make sure the backend is running on http://localhost:8000';
      } else if (error instanceof Error) {
        errorContent = `Error: ${error.message}`;
      }
      
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

  // Edit handlers
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

  // Media modal handlers
  const handleMediaClick = (url: string, type: 'image' | 'video', duration?: number) => {
    setModalState({ isOpen: true, type, url, duration });
  };

  const closeModal = () => {
    setModalState({ isOpen: false, type: 'image', url: '' });
  };

  // Key handlers
  const handleInputKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Event handling
  useEffect(() => {
    if (!lastEvent) return;
    
    const eventType = lastEvent.type;
    const eventAction = lastEvent.action;
    
    switch (eventType) {
      case 'audio_event':
        handleAudioEvent(eventAction, lastEvent.data);
        break;
      case 'llm_event':
        handleLLMEvent(eventAction, lastEvent.data);
        break;
      case 'tool_event':
        handleToolEvent(eventAction, lastEvent.data);
        break;
    }
  }, [lastEvent]);

  const handleAudioEvent = (action: string, data: Record<string, any>) => {
    switch (action) {
      case 'transcription_start':
        isTranscribing.current = true;
        break;
      case 'transcription_end':
        isTranscribing.current = false;
        break;
      case 'raw_transcript':
        const rawTranscript = data?.transcript;
        if (rawTranscript && rawTranscript.trim() && isInContextMode.current) {
          currentContext.current = currentContext.current 
            ? `${currentContext.current} ${rawTranscript}` 
            : rawTranscript;
        }
        break;
      case 'wake_word_detected':
        isInContextMode.current = true;
        currentContext.current = data?.transcript || '';
        break;
      case 'context_ready':
        const transcript = data?.transcript;
        if (transcript) {
          const lastClientMessage = messages[messages.length - 1];
          const shouldAddNewUserMessage = !(
            lastClientMessage &&
            lastClientMessage.type === 'user' &&
            lastClientMessage.content === transcript
          );

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
        
        isInContextMode.current = false;
        currentContext.current = '';
        isTranscribing.current = false;
        break;
    }
  };

  const handleLLMEvent = (action: string, data: Record<string, any>) => {
    switch (action) {
      case 'response_start':
        setCurrentResponse('');
        isProcessing.current = true;
        setCurrentResponseImageBase64(null);
        break;
      case 'response_chunk':
        const chunk = data?.chunk;
        if (chunk) {
          setCurrentResponse(prev => prev + chunk);
        }
        break;
      case 'response_end':
        const fullResponse = data?.full_response;
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
  };

  const handleToolEvent = (action: string, data: Record<string, any>) => {
    switch (action) {
      case 'video_start':
        const duration = data?.duration || 3;
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
          id: 'photo_' + Date.now().toString(),
          type: 'assistant',
          content: `📸 Capturing photo for analysis...`,
          timestamp: new Date(),
          isComplete: true
        };
        setMessages(prev => [...prev, photoMessage]);
        break;
        
      case 'photo_complete':
        const capturedPhotoData = data;
        if (capturedPhotoData?.success) {
          setMessages(prev => {
            const updated = prev.map(msg => {
              if (msg.id.startsWith('photo_') && msg.content.includes('Capturing photo for analysis')) {
                return { ...msg, content: `✅ Photo captured successfully` };
              }
              return msg;
            });
            
            if (capturedPhotoData.photo_base64) {
              const photoToolMessage: Message = {
                id: Date.now().toString(),
                type: 'tool',
                content: 'Photo captured',
                timestamp: new Date(),
                isComplete: true,
                imageBase64: capturedPhotoData.photo_base64,
                toolName: 'get_photo'
              };
              updated.push(photoToolMessage);
            }
            
            return updated;
          });
        } else {
          setMessages(prev => prev.map(msg => 
            msg.id.startsWith('photo_') && msg.content.includes('Capturing photo for analysis')
              ? { ...msg, content: `❌ Photo capture failed` }
              : msg
          ));
        }
        break;
        
      case 'video_complete':
        const videoData = data;
        if (videoData?.success) {
          setMessages(prev => prev.map(msg => 
            msg.id.startsWith('video_') && msg.content.includes('Recording') && msg.content.includes('video to analyze')
              ? { ...msg, content: `✅ Video recorded successfully (${videoData.duration}s)` }
              : msg
          ));
          
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
          setMessages(prev => prev.map(msg => 
            msg.id.startsWith('video_') && msg.content.includes('Recording') && msg.content.includes('video to analyze')
              ? { ...msg, content: `❌ Video recording failed` }
              : msg
          ));
        }
        break;
    }
  };
  
  const statusInfo = getStatusInfo();
  
  return (
    <div className={`elevated-card rounded-xl flex flex-col h-full overflow-hidden  ${className}`}>
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg flex items-center justify-center text-xs shadow-lg">
              💬
            </div>
            <h2 className="text-base font-semibold text-white">Conversation</h2>
            {isInContextMode.current && (
              <div className="flex items-center">
                <div className="status-indicator warning ml-2">
                  <span className="text-xs font-medium">Listening...</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Context Accumulation Display */}
      {isInContextMode.current && currentContext.current && (
        <div className="flex-shrink-0 mx-4 mt-3 p-3 bg-yellow-900/20 border border-yellow-700/50 rounded-xl backdrop-blur-sm">
          <div className="flex items-center mb-2">
            <div className="status-indicator warning">
              <span className="text-sm font-medium text-yellow-300">Building Context</span>
            </div>
          </div>
          <div className="text-sm text-yellow-100 bg-yellow-900/30 border border-yellow-800/30 p-2 rounded-lg font-mono leading-relaxed">
            {currentContext.current}
          </div>
          <div className="text-xs text-yellow-400 mt-1 font-medium">
            Continue speaking or wait 2 seconds to send to AI
          </div>
        </div>
      )}
      
      {/* Messages Container */}
      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.length === 0 && !isProcessing.current && !isInContextMode.current && (
            <div className="text-center py-8 px-4">
              <div className="w-12 h-12 bg-gradient-to-br from-gray-700 to-gray-800 rounded-2xl flex items-center justify-center text-xl mx-auto mb-3 shadow-lg">
                🎤
              </div>
              <h3 className="text-base font-semibold text-gray-300 mb-2">Ready to Listen</h3>
              <p className="text-gray-400 mb-2 text-sm">Say "Osmo" to start a conversation</p>
              <div className="space-y-1 text-xs text-gray-500">
                <p>Context will accumulate until 2 seconds of silence</p>
                <p>You can also type messages below ↓</p>
              </div>
            </div>
          )}
          
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              isEditing={editingMessageId === message.id}
              editContent={editingContent}
              onStartEdit={startEditingMessage}
              onSaveEdit={saveEditedMessage}
              onCancelEdit={cancelEditing}
              onEditContentChange={setEditingContent}
              onMediaClick={handleMediaClick}
            />
          ))}
          
          {/* Current streaming response */}
          {isProcessing.current && (
            <div className="flex justify-start fade-in">
              <div className="max-w-[85%] bg-gray-800/80 border border-gray-700/50 text-gray-100 rounded-xl px-3 py-2 shadow-lg backdrop-blur-sm">
                <div className="text-sm mb-1 break-words leading-relaxed">
                  {currentResponse}
                  {currentResponseImageBase64 && (
                    <img
                      src={`data:image/jpeg;base64,${currentResponseImageBase64}`}
                      alt="Current context"
                      className="mt-2 rounded-lg max-w-[120px] max-h-[120px] object-cover shadow-md border border-gray-600/50"
                    />
                  )}
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse"></div>
                  <span className="text-xs text-gray-400 font-medium">
                    {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>
      
      {/* Input Section */}
      <div className="flex-shrink-0 p-4 pt-3 border-t border-gray-800">
        <div className="flex space-x-2 mb-3">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleInputKeyPress}
            placeholder="Type a message..."
            disabled={isProcessing.current}
            className="flex-1 bg-gray-800/50 text-white border border-gray-700/50 rounded-lg px-3 py-2 text-sm placeholder-gray-500 transition-all duration-200 focus:outline-none focus:border-gray-400/50 focus:bg-gray-800/80 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isProcessing.current}
            className="bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-500 hover:to-gray-600 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 shadow-lg hover:shadow-gray-500/25 disabled:shadow-none"
          >
            Send
          </button>
        </div>
        
        {/* Status indicator */}
        <div className="flex items-center">
          <div className={`inline-flex items-center px-2 py-1 rounded-lg text-xs font-medium border transition-all duration-200 ${
            statusInfo.color === 'bg-blue-400' ? 'bg-blue-900/40 text-blue-300 border-blue-800/50' :
            statusInfo.color === 'bg-yellow-400' ? 'bg-yellow-900/40 text-yellow-300 border-yellow-800/50' :
            statusInfo.color === 'bg-orange-400' ? 'bg-orange-900/40 text-orange-300 border-orange-800/50' :
            'bg-green-900/40 text-green-300 border-green-800/50'
          }`}>
            <div className={`w-1 h-1 rounded-full mr-1 ${
              statusInfo.color === 'bg-blue-400' ? 'bg-blue-400' :
              statusInfo.color === 'bg-yellow-400' ? 'bg-yellow-400' :
              statusInfo.color === 'bg-orange-400' ? 'bg-orange-400' :
              'bg-green-400'
            } ${
              isTranscribing.current || isProcessing.current || isInContextMode.current ? 'animate-pulse' : ''
            }`} />
            {statusInfo.text}
          </div>
        </div>
      </div>

      {/* Media Modal */}
      <MediaModal
        isOpen={modalState.isOpen}
        type={modalState.type}
        url={modalState.url}
        duration={modalState.duration}
        onClose={closeModal}
      />
    </div>
  );
} 