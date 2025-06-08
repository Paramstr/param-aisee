import { useState, useEffect, useRef, useCallback } from 'react';

export interface Event {
  type: string;
  action: string;
  data: Record<string, any>;
  timestamp: string;
}

export interface SystemStatus {
  is_running: boolean;
  audio_listening: boolean;
  vision_capturing: boolean;
  llm_processing: boolean;
  whisper_loaded: boolean;
}

export interface ToolState {
  active: boolean;
  tool_name: string | null;
  action: string | null;
  message: string | null;
  duration?: number;
}

interface UseSocketReturn {
  isConnected: boolean;
  lastEvent: Event | null;
  systemStatus: SystemStatus | null;
  toolState: ToolState;
  error: string | null;
}

export function useSocket(url: string): UseSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<Event | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [toolState, setToolState] = useState<ToolState>({
    active: false,
    tool_name: null,
    action: null,
    message: null,
  });
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = useRef(1000);

  const connect = useCallback(() => {
    try {
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
        reconnectDelay.current = 1000;
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('WebSocket message received:', message);
          
          // Handle different message types from backend
          switch (message.type) {
            case 'event':
              // This is the main event wrapper from backend
              if (message.event && message.event.type && message.event.action) {
                const newEvent: Event = {
                  type: message.event.type,
                  action: message.event.action,
                  data: message.event.data || {},
                  timestamp: message.event.timestamp || new Date().toISOString(),
                };
                
                setLastEvent(newEvent);
                console.log('Processed event:', newEvent);

                // Handle specific event types
                handleEventByType(newEvent);
              }
              break;
              
            case 'status':
              // Initial status message
              if (message.data) {
                setSystemStatus(message.data);
                console.log('System status updated:', message.data);
              }
              break;
              
            case 'ping':
              // Heartbeat from server, no action needed
              break;
              
            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err, 'Raw message:', event.data);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        
        // Attempt to reconnect if not manually closed
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          console.log(`Reconnect attempt ${reconnectAttempts.current}/${maxReconnectAttempts}`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay.current);
          
          // Exponential backoff
          reconnectDelay.current = Math.min(reconnectDelay.current * 2, 30000);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setError('Failed to reconnect to WebSocket after multiple attempts');
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection error');
      };

    } catch (err) {
      console.error('Error creating WebSocket:', err);
      setError('Failed to create WebSocket connection');
    }
  }, [url]);

  const handleEventByType = (event: Event) => {
    switch (event.type) {
      case 'system_status':
        // Update system status from event data
        if (event.data?.status) {
          setSystemStatus(event.data.status);
        }
        break;
        
      case 'tool_event':
        handleToolEvent(event.action, event.data);
        break;
        
      // Add other event type handlers as needed
      case 'audio_event':
      case 'llm_event':
      case 'tts_event':
      case 'vision_event':
      case 'error':
        // These are handled by components directly via lastEvent
        break;
        
      default:
        console.log('Unhandled event type:', event.type);
    }
  };

  const handleToolEvent = (action: string, data: Record<string, unknown>) => {
    console.log('🔧 handleToolEvent called:', { action, data });
    
    switch (action) {
      case 'photo_start':
        console.log('📸 Setting photo_start state');
        setToolState({
          active: true,
          tool_name: 'get_photo',
          action: 'starting',
          message: 'Preparing camera...',
        });
        // Auto-clear after 10 seconds if no completion event
        setTimeout(() => {
          console.log('📸 Auto-clearing photo state after 10s timeout');
          setToolState(prev => prev.active && prev.tool_name === 'get_photo' ? {
            active: false,
            tool_name: null,
            action: null,
            message: null,
          } : prev);
        }, 10000);
        break;
        
      case 'photo_capture':
        console.log('📸 Setting photo_capture state');
        setToolState({
          active: true,
          tool_name: 'get_photo',
          action: 'capturing',
          message: 'Capturing photo...',
        });
        break;
        
      case 'photo_complete':
        console.log('📸 Setting photo_complete state');
        setToolState({
          active: false,
          tool_name: 'get_photo',
          action: 'complete',
          message: data.success ? 'Photo captured successfully' : 'Photo capture failed',
        });
        // Clear after 3 seconds
        setTimeout(() => {
          console.log('📸 Clearing photo state after 3s');
          setToolState({
            active: false,
            tool_name: null,
            action: null,
            message: null,
          });
        }, 3000);
        break;
        
      case 'video_start':
        console.log('🎥 Setting video_start state:', { duration: data.duration, tool: data.tool });
        setToolState({
          active: true,
          tool_name: data.tool as string || 'get_video',
          action: 'starting',
          message: data.message as string || `Starting ${data.duration}s video recording...`,
          duration: data.duration as number,
        });
        break;
        
      case 'video_recording':
        console.log('🎥 Setting video_recording state:', { duration: data.duration });
        setToolState(prev => ({
          active: true,
          tool_name: prev.tool_name || 'get_video', // Preserve tool_name from previous state
          action: 'recording',
          message: data.message as string || `Recording ${data.duration}s video...`,
          duration: data.duration as number,
        }));
        break;
        
      case 'video_complete':
        console.log('🎥 Setting video_complete state:', { success: data.success });
        setToolState(prev => ({
          active: false,
          tool_name: prev.tool_name || 'get_video', // Preserve tool_name for completion display
          action: 'complete',
          message: data.message as string || (data.success ? 'Video recorded successfully' : 'Video recording failed'),
          duration: prev.duration, // Preserve duration for completion display
        }));
        // Clear after 3 seconds
        setTimeout(() => {
          console.log('🎥 Clearing video state after 3s');
          setToolState({
            active: false,
            tool_name: null,
            action: null,
            message: null,
          });
        }, 3000);
        break;
        
      case 'video_error':
        console.log('🎥 Setting video_error state:', { error: data.error });
        setToolState(prev => ({
          active: false,
          tool_name: prev.tool_name || 'get_video',
          action: 'error',
          message: data.message as string || `Video error: ${data.error}`,
          duration: prev.duration,
        }));
        // Clear after 5 seconds for errors
        setTimeout(() => {
          console.log('🎥 Clearing video error state after 5s');
          setToolState({
            active: false,
            tool_name: null,
            action: null,
            message: null,
          });
        }, 5000);
        break;
        
      default:
        console.log('❓ Unhandled tool event action:', action);
    }
  };

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
      }
    };
  }, [connect]);

  return {
    isConnected,
    lastEvent,
    systemStatus,
    toolState,
    error,
  };
} 