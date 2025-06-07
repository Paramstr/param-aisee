import React, { useState } from 'react';

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

interface ChatMessageProps {
  message: Message;
  isEditing: boolean;
  editContent: string;
  onStartEdit: (id: string, content: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
  onEditContentChange: (content: string) => void;
  onMediaClick: (url: string, type: 'image' | 'video', duration?: number) => void;
}

export function ChatMessage({
  message,
  isEditing,
  editContent,
  onStartEdit,
  onSaveEdit,
  onCancelEdit,
  onEditContentChange,
  onMediaClick
}: ChatMessageProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const handleEditKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSaveEdit();
    } else if (e.key === 'Escape') {
      onCancelEdit();
    }
  };

  return (
    <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} fade-in`}>
      <div className={`max-w-[85%] rounded-xl px-4 py-3 group relative shadow-lg transition-all duration-200 ${
        message.type === 'user'
          ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white border border-blue-500/20'
          : message.type === 'tool'
          ? 'bg-gradient-to-br from-purple-600 to-purple-700 text-white border border-purple-500/20 shadow-purple-500/20'
          : 'bg-gray-800/80 text-gray-100 border border-gray-700/50 backdrop-blur-sm'
      }`}>
        {isEditing ? (
          <div className="space-y-3">
            <textarea
              value={editContent}
              onChange={(e) => onEditContentChange(e.target.value)}
              onKeyDown={handleEditKeyPress}
              className="w-full bg-gray-900/50 text-white p-3 rounded-lg text-sm resize-none min-w-[200px] border border-gray-600/50 focus:border-blue-500/50 focus:outline-none transition-colors"
              rows={3}
              autoFocus
            />
            <div className="flex space-x-2">
              <button
                onClick={onSaveEdit}
                className="text-xs bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white px-3 py-1.5 rounded-lg font-medium transition-all duration-200"
              >
                Save
              </button>
              <button
                onClick={onCancelEdit}
                className="text-xs bg-gray-600/80 hover:bg-gray-500/80 text-white px-3 py-1.5 rounded-lg font-medium transition-all duration-200"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Tool Messages */}
            {message.type === 'tool' && (
              <ToolMessageContent 
                message={message} 
                onMediaClick={onMediaClick}
              />
            )}
            
            {/* Regular Messages */}
            {message.type !== 'tool' && (
              <RegularMessageContent 
                message={message} 
                onMediaClick={onMediaClick}
              />
            )}

            {/* Message Footer */}
            <div className="flex items-center justify-between mt-2 pt-2 border-t border-white/10">
              <div className={`text-xs font-medium opacity-70 ${
                message.type === 'user' ? 'text-blue-200' : 
                message.type === 'tool' ? 'text-purple-200' :
                'text-gray-400'
              }`}>
                {formatTime(message.timestamp)}
              </div>
              {message.type === 'user' && (
                <button
                  onClick={() => onStartEdit(message.id, message.content)}
                  className="text-xs opacity-0 group-hover:opacity-100 transition-all duration-200 hover:bg-white/10 px-2 py-1 rounded-lg"
                >
                  ✏️
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function ToolMessageContent({ 
  message, 
  onMediaClick 
}: { 
  message: Message; 
  onMediaClick: (url: string, type: 'image' | 'video', duration?: number) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 text-sm font-medium">
        <div className="w-6 h-6 bg-white/20 rounded-lg flex items-center justify-center">
          <span>{message.toolName === 'get_video' ? '🎥' : '📸'}</span>
        </div>
        <span>{message.content}</span>
      </div>
      
      {/* Video Content */}
      {message.videoBase64 && (
        <div className="mt-3">
          <div 
            className="relative bg-black/50 rounded-lg overflow-hidden cursor-pointer hover:bg-black/40 transition-all duration-200 shadow-lg"
            style={{ aspectRatio: '16/9', width: '200px' }}
            onClick={() => onMediaClick(`data:video/mp4;base64,${message.videoBase64}`, 'video', message.duration)}
          >
            <video
              className="w-full h-full object-cover"
              preload="metadata"
            >
              <source src={`data:video/mp4;base64,${message.videoBase64}`} type="video/mp4" />
            </video>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-black/60 backdrop-blur-sm rounded-full p-3 shadow-lg">
                <div className="w-8 h-8 flex items-center justify-center text-white text-xl">▶</div>
              </div>
            </div>
          </div>
          <div className="text-xs text-purple-200 mt-2 font-medium">
            Duration: {message.duration}s • Click to play
          </div>
        </div>
      )}
      
      {/* Photo Content */}
      {message.imageBase64 && (
        <div className="mt-3">
          <img
            src={`data:image/jpeg;base64,${message.imageBase64}`}
            alt="Captured photo"
            className="rounded-lg object-cover cursor-pointer hover:opacity-80 transition-all duration-200 shadow-lg border border-white/10"
            style={{ maxWidth: '150px', maxHeight: '150px' }}
            onClick={() => onMediaClick(`data:image/jpeg;base64,${message.imageBase64}`, 'image')}
          />
          <div className="text-xs text-purple-200 mt-2 font-medium">
            Click to enlarge
          </div>
        </div>
      )}
    </div>
  );
}

function RegularMessageContent({ 
  message, 
  onMediaClick 
}: { 
  message: Message; 
  onMediaClick: (url: string, type: 'image' | 'video', duration?: number) => void;
}) {
  if (message.type === 'user' && message.imageBase64) {
    return (
      <div className="flex items-start space-x-3">
        <img
          src={`data:image/jpeg;base64,${message.imageBase64}`}
          alt="User context"
          className="rounded-lg object-cover cursor-pointer hover:opacity-80 transition-all duration-200 flex-shrink-0 shadow-md border border-white/10"
          style={{ width: '48px', height: '48px' }}
          onClick={() => onMediaClick(`data:image/jpeg;base64,${message.imageBase64}`, 'image')}
        />
        <div className="text-sm flex-1 min-w-0 leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="text-sm break-words leading-relaxed">
        {message.content}
      </div>
      {message.type === 'assistant' && message.imageBase64 && (
        <img
          src={`data:image/jpeg;base64,${message.imageBase64}`}
          alt="Assistant context"
          className="mt-3 rounded-lg object-cover cursor-pointer hover:opacity-80 transition-all duration-200 shadow-md border border-gray-600/50"
          style={{ maxWidth: '120px', maxHeight: '120px' }}
          onClick={() => onMediaClick(`data:image/jpeg;base64,${message.imageBase64}`, 'image')}
        />
      )}
    </>
  );
} 