import { useState, useEffect, useRef, useCallback } from 'react';

export interface Event {
  type: string;
  action: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface SystemStatus {
  is_running: boolean;
  audio_listening: boolean;
  voice_dictation_enabled: boolean;
  vision_capturing: boolean;
  camera_capture_enabled: boolean;
  llm_processing: boolean;
  whisper_loaded: boolean;
  tts_enabled: boolean;
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
  refreshSystemStatus: () => void;
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

  const refreshSystemStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status');
      if (response.ok) {
        const status = await response.json();
        setSystemStatus(status);
      }
    } catch (err) {
      console.error('Failed to refresh system status:', err);
    }
  }, []);

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
          setSystemStatus(event.data.status as SystemStatus);
        }
        break;
        
      case 'tool_event':
        handleToolEvent(event.action, event.data);
        // Also let components handle tool events via lastEvent
        break;
        
      case 'voice_control':
        // Voice dictation events - refresh system status
        if (event.action === 'dictation_toggled') {
          // Update the system status locally to reflect the change immediately
          if (systemStatus) {
            const updatedStatus = {
              ...systemStatus,
              voice_dictation_enabled: event.data.enabled as boolean
            };
            setSystemStatus(updatedStatus);
          }
        }
        break;
        
      case 'camera_control':
        // Camera capture events - refresh system status
        if (event.action === 'capture_toggled') {
          // Update the system status locally to reflect the change immediately
          if (systemStatus) {
            const updatedStatus = {
              ...systemStatus,
              camera_capture_enabled: event.data.enabled as boolean
            };
            setSystemStatus(updatedStatus);
          }
        }
        break;
        
      case 'tts_control':
        // TTS events - refresh system status
        if (event.action === 'tts_toggled') {
          // Update the system status locally to reflect the change immediately
          if (systemStatus) {
            const updatedStatus = {
              ...systemStatus,
              tts_enabled: event.data.enabled as boolean
            };
            setSystemStatus(updatedStatus);
          }
        }
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
      case 'tool_start':
        console.log('🔧 Tool start event for:', data.tool);
        const toolName = data.tool as string;
        // Skip showing UI for photo tool_start - only show photo_complete
        if (toolName === 'get_photo') {
          console.log('📸 Ignoring tool_start for get_photo - only showing completion');
          break;
        }
        setToolState({
          active: true,
          tool_name: toolName,
          action: 'starting',
          message: `Starting ${toolName}...`,
        });
        // Auto-clear after 10 seconds if no completion event
        setTimeout(() => {
          console.log('🔧 Auto-clearing tool state after 10s timeout');
          setToolState(prev => prev.active && prev.tool_name === toolName ? {
            active: false,
            tool_name: null,
            action: null,
            message: null,
          } : prev);
        }, 10000);
        break;
        
      case 'photo_start':
        console.log('📸 Ignoring photo_start - only showing completion');
        // Skip showing UI for photo_start - only show photo_complete
        break;
        
      case 'photo_capture':
        console.log('📸 Ignoring photo_capture - only showing completion');
        // Skip showing UI for photo_capture - only show photo_complete
        break;
        
      case 'photo_complete':
        console.log('📸 Showing photo completion');
        setToolState({
          active: true,
          tool_name: 'get_photo',
          action: 'complete',
          message: 'Photo captured',  // Simple, clean message
        });
        // Show for 2 seconds, then clear completely
        setTimeout(() => {
          console.log('📸 Clearing photo state after 2s');
          setToolState({
            active: false,
            tool_name: null,
            action: null,
            message: null,
          });
        }, 2000);
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
        
      case 'tool_complete':
        console.log('🔧 Setting tool_complete state for:', data.tool);
        const completedToolName = data.tool as string;
        
        // Don't override photo/video completion state - let specific handlers manage their own timing
        if (completedToolName === 'get_photo') {
          console.log('📸 Ignoring tool_complete for get_photo - letting photo_complete handle timing');
          break;
        }
        if (completedToolName === 'get_video') {
          console.log('🎥 Ignoring tool_complete for get_video - letting video_complete handle timing');
          break;
        }
        
        setToolState(prev => ({
          active: false,
          tool_name: prev.tool_name || completedToolName,
          action: 'complete',
          message: data.success ? `${completedToolName} completed successfully` : `${completedToolName} failed`,
        }));
        // Clear after 3 seconds
        setTimeout(() => {
          console.log('🔧 Clearing tool complete state after 3s');
          setToolState({
            active: false,
            tool_name: null,
            action: null,
            message: null,
          });
        }, 3000);
        break;
        
      case 'tool_error':
        console.log('🔧 Setting tool_error state for:', data.tool);
        const errorToolName = data.tool as string;
        setToolState(prev => ({
          active: false,
          tool_name: prev.tool_name || errorToolName,
          action: 'error',
          message: data.error as string || `${errorToolName} failed`,
        }));
        // Clear after 5 seconds for errors
        setTimeout(() => {
          console.log('🔧 Clearing tool error state after 5s');
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
    refreshSystemStatus,
  };
} 