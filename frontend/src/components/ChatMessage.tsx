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
    <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] rounded-lg px-3 py-2 group relative ${
        message.type === 'user'
          ? 'bg-blue-600 text-white'
          : message.type === 'tool'
          ? 'bg-purple-600 text-white border-l-4 border-purple-400'
          : 'bg-gray-700 text-gray-100'
      }`}>
        {isEditing ? (
          <div className="space-y-2">
            <textarea
              value={editContent}
              onChange={(e) => onEditContentChange(e.target.value)}
              onKeyDown={handleEditKeyPress}
              className="w-full bg-gray-800 text-white p-2 rounded text-sm resize-none min-w-[200px]"
              rows={3}
              autoFocus
            />
            <div className="flex space-x-2">
              <button
                onClick={onSaveEdit}
                className="text-xs bg-green-600 hover:bg-green-700 text-white px-2 py-1 rounded"
              >
                Save
              </button>
              <button
                onClick={onCancelEdit}
                className="text-xs bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 rounded"
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
            <div className="flex items-center justify-between mt-1">
              <div className={`text-xs opacity-70 ${
                message.type === 'user' ? 'text-blue-200' : 
                message.type === 'tool' ? 'text-purple-200' :
                'text-gray-400'
              }`}>
                {formatTime(message.timestamp)}
              </div>
              {message.type === 'user' && (
                <button
                  onClick={() => onStartEdit(message.id, message.content)}
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
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-sm font-medium">
        <span>{message.toolName === 'get_video' ? '🎥' : '📸'}</span>
        <span>{message.content}</span>
      </div>
      
      {/* Video Content */}
      {message.videoBase64 && (
        <div className="mt-2">
          <div 
            className="relative bg-black rounded-lg overflow-hidden cursor-pointer hover:opacity-90 transition-opacity"
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
              <div className="bg-black bg-opacity-50 rounded-full p-2">
                <div className="w-8 h-8 flex items-center justify-center text-white text-xl">▶</div>
              </div>
            </div>
          </div>
          <div className="text-xs text-purple-200 mt-1">
            Duration: {message.duration}s • Click to play
          </div>
        </div>
      )}
      
      {/* Photo Content */}
      {message.imageBase64 && (
        <div className="mt-2">
          <img
            src={`data:image/jpeg;base64,${message.imageBase64}`}
            alt="Captured photo"
            className="rounded-lg object-cover cursor-pointer hover:opacity-80 transition-opacity"
            style={{ maxWidth: '150px', maxHeight: '150px' }}
            onClick={() => onMediaClick(`data:image/jpeg;base64,${message.imageBase64}`, 'image')}
          />
          <div className="text-xs text-purple-200 mt-1">
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
      <div className="flex items-start space-x-2">
        <img
          src={`data:image/jpeg;base64,${message.imageBase64}`}
          alt="User context"
          className="rounded-md object-cover cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
          style={{ width: '48px', height: '48px' }}
          onClick={() => onMediaClick(`data:image/jpeg;base64,${message.imageBase64}`, 'image')}
        />
        <div className="text-sm flex-1 min-w-0">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="text-sm break-words">
        {message.content}
      </div>
      {message.type === 'assistant' && message.imageBase64 && (
        <img
          src={`data:image/jpeg;base64,${message.imageBase64}`}
          alt="Assistant context"
          className="mt-2 rounded-md object-cover cursor-pointer hover:opacity-80 transition-opacity"
          style={{ maxWidth: '120px', maxHeight: '120px' }}
          onClick={() => onMediaClick(`data:image/jpeg;base64,${message.imageBase64}`, 'image')}
        />
      )}
    </>
  );
} 