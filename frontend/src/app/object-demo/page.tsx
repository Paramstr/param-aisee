'use client';

import { useState, useEffect, useRef } from 'react';
import { useWebSocket } from '@/lib/WebSocketProvider';
import { CameraFeed } from '@/components/CameraFeed';

interface DetectionResult {
  id: string;
  timestamp: string;
  videoTimestamp?: number;
  latency: number;
  frameUrl: string;
  busNumber: string;
  confidence?: number;
  frameIndex: number;
}

interface BusVideo {
  id: string;
  name: string;
  description: string;
  duration: number;
  thumbnail: string;
  url?: string;
  type?: string;
  filename?: string;
}

export default function BusDemo() {
  const [busVideos, setBusVideos] = useState<BusVideo[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<BusVideo | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>([]);
  const [totalLatency, setTotalLatency] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [useCloud, setUseCloud] = useState(false);
  const [cloudAvailable, setCloudAvailable] = useState(false);
  const [localAvailable, setLocalAvailable] = useState(false);
  const [expandedFrame, setExpandedFrame] = useState<DetectionResult | null>(null);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [isEditingPrompt, setIsEditingPrompt] = useState(false);
  const [promptText, setPromptText] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [detectionMode, setDetectionMode] = useState<'video' | 'camera'>('video');
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { lastEvent } = useWebSocket();

  // Fetch available videos and status from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch videos
        const videosResponse = await fetch('http://localhost:8000/object-demo/videos');
        if (videosResponse.ok) {
          const videosData = await videosResponse.json();
          const videos = videosData.videos || [];
          
          // Add video URLs for each video
          const videosWithUrls = await Promise.all(
            videos.map(async (video: BusVideo) => {
              try {
                const videoInfoResponse = await fetch(`http://localhost:8000/object-demo/video/${video.id}`);
                if (videoInfoResponse.ok) {
                  const videoInfo = await videoInfoResponse.json();
                  return {
                    ...video,
                    url: `http://localhost:8000${videoInfo.url}`,
                    filename: videoInfo.filename || video.filename
                  };
                }
              } catch (e) {
                console.warn(`Failed to get info for video ${video.id}:`, e);
              }
              return video;
            })
          );
          
          setBusVideos(videosWithUrls);
        }
        
        // Fetch status
        const statusResponse = await fetch('http://localhost:8000/object-demo/status');
        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          setCloudAvailable(statusData.cloud_available || false);
          setLocalAvailable(statusData.local_available || false);
          setUseCloud(statusData.current_mode === 'cloud');
          setSystemPrompt(statusData.system_prompt || '');
          setPromptText(statusData.system_prompt || '');
        }
      } catch (error) {
        console.error('Failed to fetch data:', error);
      }
    };
    fetchData();
  }, []);

  // Handle WebSocket events for real-time detection results
  useEffect(() => {
    if (!lastEvent) return;

    console.log(`🚌 Bus demo received event:`, lastEvent);

    if (lastEvent.type === 'object_demo') {
      console.log(`🎯 Processing object_demo event: ${lastEvent.action}`);
      switch (lastEvent.action) {
        case 'detection_started':
          setIsDetecting(true);
          setDetectionResults([]);
          setTotalLatency(0);
          setFrameCount(0);
          break;
          
        case 'detection_result':
          const result = lastEvent.data as {
            timestamp: string;
            videoTimestamp?: number;
            latency: number;
            frameUrl: string;
            detectedObjects: string;
            frameIndex: number;
          };
          const newResult: DetectionResult = {
            id: `${Date.now()}-${Math.random()}`,
            timestamp: result.timestamp || '',
            videoTimestamp: result.videoTimestamp,
            latency: result.latency || 0,
            frameUrl: result.frameUrl || '',
            busNumber: result.detectedObjects || '',
            confidence: 0.85, // Default confidence
            frameIndex: result.frameIndex || 0
          };
          
          setDetectionResults(prev => [...prev, newResult]);
          setTotalLatency(prev => prev + (result.latency || 0));
          setFrameCount(prev => prev + 1);
          break;
          
        case 'detection_completed':
        case 'detection_stopped':
          setIsDetecting(false);
          break;
          
        case 'detection_error':
          setIsDetecting(false);
          console.error('Detection error:', lastEvent.data?.error);
          break;
      }
    }
  }, [lastEvent]);

  // Clear detection results when switching detection modes
  useEffect(() => {
    setDetectionResults([]);
    setTotalLatency(0);
    setFrameCount(0);
  }, [detectionMode]);

  const startDetection = async () => {
    if (detectionMode === 'video') {
      if (!selectedVideo) {
        console.warn('❌ No video selected for detection');
        return;
      }
      
      console.log(`🎬 Starting video detection for: ${selectedVideo.id} (${selectedVideo.name})`);
      
      try {
        const response = await fetch(`http://localhost:8000/object-demo/start/${selectedVideo.id}`, {
          method: 'POST'
        });
        
        if (response.ok) {
          const result = await response.json();
          console.log('✅ Video detection started:', result.message);
          // State will be updated via WebSocket events
        } else {
          const error = await response.json();
          console.error('❌ Failed to start video detection:', error.detail);
          alert(`Failed to start video detection: ${error.detail}`);
        }
      } catch (error) {
        console.error('❌ Error starting video detection:', error);
        alert('Error starting video detection. Make sure the backend is running.');
      }
    } else {
      // Camera mode
      console.log('📹 Starting real-time camera detection');
      
      try {
        const response = await fetch('http://localhost:8000/object-demo/start-realtime', {
          method: 'POST'
        });
        
        if (response.ok) {
          const result = await response.json();
          console.log('✅ Camera detection started:', result.message);
          // State will be updated via WebSocket events
        } else {
          const error = await response.json();
          console.error('❌ Failed to start camera detection:', error.detail);
          alert(`Failed to start camera detection: ${error.detail}`);
        }
      } catch (error) {
        console.error('❌ Error starting camera detection:', error);
        alert('Error starting camera detection. Make sure the backend is running.');
      }
    }
  };

  const stopDetection = async () => {
    try {
      const response = await fetch('http://localhost:8000/object-demo/stop', {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Detection stopped:', result.message);
        // State will be updated via WebSocket events
      } else {
        const error = await response.json();
        console.error('Failed to stop detection:', error.detail);
      }
    } catch (error) {
      console.error('Error stopping detection:', error);
    }
  };

  const selectVideo = (video: BusVideo) => {
    setSelectedVideo(video);
    setDetectionResults([]);
  };

  const toggleInferenceMode = async () => {
    const newMode = !useCloud;
    try {
      const response = await fetch('http://localhost:8000/object-demo/inference-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_cloud: newMode })
      });
      
      if (response.ok) {
        setUseCloud(newMode);
        console.log(`Switched to ${newMode ? 'cloud' : 'local'} inference`);
      } else {
        const error = await response.json();
        alert(`Failed to switch mode: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error switching inference mode:', error);
      alert('Error switching inference mode');
    }
  };

  const updateSystemPrompt = async () => {
    try {
      const response = await fetch('http://localhost:8000/object-demo/system-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: promptText })
      });
      
      if (response.ok) {
        const result = await response.json();
        setSystemPrompt(promptText);
        setIsEditingPrompt(false);
        console.log('✅ System prompt updated:', result.message);
      } else {
        const error = await response.json();
        alert(`Failed to update system prompt: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error updating system prompt:', error);
      alert('Error updating system prompt');
    }
  };

  const updateSystemPromptWithText = async (prompt: string) => {
    try {
      const response = await fetch('http://localhost:8000/object-demo/system-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      
      if (response.ok) {
        const result = await response.json();
        setSystemPrompt(prompt);
        setPromptText(prompt);
        console.log('✅ System prompt updated:', result.message);
      } else {
        const error = await response.json();
        alert(`Failed to update system prompt: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error updating system prompt:', error);
      alert('Error updating system prompt');
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.mp4') && !file.name.toLowerCase().endsWith('.mov')) {
      alert('Please select an MP4 or MOV file');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/object-demo/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ Video uploaded successfully:', result);
        
        // Refresh video list with URLs
        const videosResponse = await fetch('http://localhost:8000/object-demo/videos');
        if (videosResponse.ok) {
          const videosData = await videosResponse.json();
          const videos = videosData.videos || [];
          
                     // Add video URLs for each video (including the newly uploaded one)
           const videosWithUrls = await Promise.all(
             videos.map(async (video: BusVideo) => {
               try {
                 const videoInfoResponse = await fetch(`http://localhost:8000/object-demo/video/${video.id}`);
                 if (videoInfoResponse.ok) {
                   const videoInfo = await videoInfoResponse.json();
                   return {
                     ...video,
                     url: `http://localhost:8000${videoInfo.url}`,
                     filename: videoInfo.filename || video.filename
                   };
                 }
               } catch (e) {
                 console.warn(`Failed to get info for video ${video.id}:`, e);
               }
               return video;
             })
           );
          
          setBusVideos(videosWithUrls);
        }
        
        alert(`Video uploaded successfully! Duration: ${result.duration}s`);
      } else {
        const error = await response.json();
        alert(`Failed to upload video: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error uploading video:', error);
      alert('Error uploading video. Make sure the backend is running.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const averageLatency = frameCount > 0 ? totalLatency / frameCount : 0;

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden bg-gradient-to-br">
      <div className="p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-6">
            <div className="text-6xl mb-4">🚌</div>
            <h1 className="text-4xl font-bold text-white mb-2">Object Detection Demo</h1>
            <p className="text-gray-400 text-lg mb-6">
              Simulate realtime object detection using Moondream VLM with latency measurements
            </p>
            
            {/* Detection Controls at Top */}
            <div className="max-w-md mx-auto">
              <div className="relative group">
                <button
                  onClick={startDetection}
                  disabled={(detectionMode === 'video' && !selectedVideo) || isDetecting}
                  className={`w-full py-4 px-6 rounded-xl font-medium text-lg transition-all duration-200 ${
                    (detectionMode === 'video' && !selectedVideo) || isDetecting
                      ? 'bg-gray-700/50 text-gray-500 cursor-not-allowed'
                      : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-green-600/25'
                  }`}
                >
                  {isDetecting ? (
                    <div className="flex items-center justify-center space-x-2">
                      <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                      <span>Detecting...</span>
                    </div>
                  ) : (
                    '🔍 Watch for Objects'
                  )}
                </button>
                
                {/* Tooltip for disabled state */}
                {((detectionMode === 'video' && !selectedVideo) || isDetecting) && (
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10 border border-gray-700">
                    {isDetecting 
                      ? 'Detection is already running'
                      : detectionMode === 'video' && !selectedVideo
                      ? 'Please select a video scenario first'
                      : 'Unable to start detection'
                    }
                    {/* Tooltip arrow */}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                  </div>
                )}
              </div>
              
              {isDetecting && (
                <button
                  onClick={stopDetection}
                  className="w-full mt-3 py-2 px-4 rounded-lg font-medium bg-red-600 hover:bg-red-700 text-white transition-all duration-200"
                >
                  Stop Detection
                </button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Left Panel - Compact Configuration */}
            <div className="xl:col-span-1 space-y-4">
              {/* Detection Mode & Inference Mode Combined */}
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                <h2 className="text-lg font-semibold text-white mb-3">Configuration</h2>
                
                {/* Detection Mode */}
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-400 mb-2">Detection Mode</h3>
                  <div className="flex bg-gray-700/30 rounded-lg p-1">
                    <button
                      onClick={() => setDetectionMode('video')}
                      className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                        detectionMode === 'video'
                          ? 'bg-blue-600 text-white shadow-lg'
                          : 'text-gray-300 hover:text-white hover:bg-gray-600/50'
                      }`}
                    >
                      <div className="flex items-center justify-center space-x-1">
                        <span>🎥</span>
                        <span>Video</span>
                      </div>
                    </button>
                    <button
                      onClick={() => setDetectionMode('camera')}
                      className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                        detectionMode === 'camera'
                          ? 'bg-blue-600 text-white shadow-lg'
                          : 'text-gray-300 hover:text-white hover:bg-gray-600/50'
                      }`}
                    >
                      <div className="flex items-center justify-center space-x-1">
                        <span>📹</span>
                        <span>Camera</span>
                      </div>
                    </button>
                  </div>
                </div>

                {/* Inference Mode */}
                <div>
                  <h3 className="text-sm font-medium text-gray-400 mb-2">Inference Mode</h3>
                  <div className="flex bg-gray-700/30 rounded-lg p-1 mb-2">
                    <button
                      onClick={() => !isDetecting && cloudAvailable && toggleInferenceMode()}
                      disabled={!cloudAvailable || isDetecting}
                      className={`flex-1 px-3 py-2 rounded-md text-xs font-medium transition-all duration-200 ${
                        useCloud
                          ? 'bg-blue-600 text-white shadow-lg'
                          : cloudAvailable && !isDetecting
                          ? 'text-gray-300 hover:text-white hover:bg-gray-600/50'
                          : 'text-gray-500 cursor-not-allowed'
                      }`}
                    >
                      <div className="text-center">
                        <div className="text-sm">☁️ Cloud</div>
                        <div className={`text-xs ${cloudAvailable ? 'text-green-400' : 'text-red-400'}`}>
                          {cloudAvailable ? '✅' : '❌'}
                        </div>
                      </div>
                    </button>
                    <button
                      onClick={() => !isDetecting && localAvailable && !useCloud ? null : toggleInferenceMode()}
                      disabled={!localAvailable || isDetecting}
                      className={`flex-1 px-3 py-2 rounded-md text-xs font-medium transition-all duration-200 ${
                        !useCloud
                          ? 'bg-green-600 text-white shadow-lg'
                          : localAvailable && !isDetecting
                          ? 'text-gray-300 hover:text-white hover:bg-gray-600/50'
                          : 'text-gray-500 cursor-not-allowed'
                      }`}
                    >
                      <div className="text-center">
                        <div className="text-sm">🖥️ Local</div>
                        <div className={`text-xs ${localAvailable ? 'text-green-400' : 'text-red-400'}`}>
                          {localAvailable ? '✅' : '❌'}
                        </div>
                      </div>
                    </button>
                  </div>
                  
                  {/* Current Mode Indicator */}
                  <div className="text-center p-2 bg-gray-700/20 rounded-lg border border-gray-600/30">
                    <div className={`text-sm font-semibold ${useCloud ? 'text-blue-400' : 'text-green-400'}`}>
                      {useCloud ? '☁️ Cloud Mode' : '🖥️ Local Mode'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Video Selection */}
              {detectionMode === 'video' && (
                <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-semibold text-white">Select Scenario</h2>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isUploading}
                      className="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded transition-colors flex items-center space-x-1"
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      <span>{isUploading ? 'Uploading...' : 'Upload'}</span>
                    </button>
                  </div>
                  <div className="space-y-2">
                    {busVideos.length > 0 ? (
                      busVideos.map((video) => (
                        <button
                          key={video.id}
                          onClick={() => selectVideo(video)}
                          className={`w-full p-3 rounded-lg border transition-all duration-200 text-left ${
                            selectedVideo?.id === video.id
                              ? 'bg-blue-600/30 border-blue-500/50 text-blue-200'
                              : 'bg-gray-700/30 border-gray-600/50 text-gray-300 hover:bg-gray-600/30'
                          }`}
                        >
                          <div className="flex items-center space-x-2">
                            <span className="text-lg">{video.thumbnail}</span>
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-sm truncate">{video.name}</div>
                              <div className="text-xs opacity-75 truncate">{video.description}</div>
                              <div className="flex items-center space-x-2 mt-1">
                                <span className="text-xs opacity-60">{video.duration}s</span>
                                {video.type === 'uploaded' && (
                                  <span className="px-1.5 py-0.5 text-xs bg-green-600/20 text-green-400 rounded">
                                    Custom
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="text-center py-4 text-gray-500">
                        <div className="text-2xl mb-1">📹</div>
                        <div className="text-xs">No videos found</div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Detection Prompt - Compact */}
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                <h2 className="text-lg font-semibold text-white mb-3">Detection Prompt</h2>

                {/* Preset Prompts */}
                <div className="mb-3">
                  <h3 className="text-xs font-medium text-gray-400 mb-2">Quick Presets:</h3>
                  <div className="space-y-1">
                    <button
                      onClick={() => {
                        const busPrompt = "Identify any bus numbers visible and return only the bus number you can see. If no bus numbers are visible, respond with 'Null'.";
                        setPromptText(busPrompt);
                        updateSystemPromptWithText(busPrompt);
                      }}
                      className={`w-full px-2 py-1.5 text-xs text-left rounded border transition-colors ${
                        systemPrompt.includes('bus number') 
                          ? 'bg-blue-600/20 border-blue-500/50 text-blue-300' 
                          : 'bg-gray-700/30 border-gray-600/50 text-gray-300 hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="flex items-center space-x-1">
                        <span>🚌</span>
                        <span className="font-medium">Bus Numbers</span>
                      </div>
                    </button>
                    
                    <button
                      onClick={() => {
                        const airpodPrompt = "Look at this image and identify if there are any AirPods or earbuds on a table or surface. Return 'AirPods detected' if you see them, or 'Null'.";
                        setPromptText(airpodPrompt);
                        updateSystemPromptWithText(airpodPrompt);
                      }}
                      className={`w-full px-2 py-1.5 text-xs text-left rounded border transition-colors ${
                        systemPrompt.includes('AirPods') 
                          ? 'bg-blue-600/20 border-blue-500/50 text-blue-300' 
                          : 'bg-gray-700/30 border-gray-600/50 text-gray-300 hover:bg-gray-600/30'
                      }`}
                    >
                      <div className="flex items-center space-x-1">
                        <span>🎧</span>
                        <span className="font-medium">AirPods Detection</span>
                      </div>
                    </button>
                  </div>
                </div>

                {/* Active Prompt Display */}
                <div>
                  <h3 className="text-xs font-medium text-gray-400 mb-2">Active Prompt:</h3>
                  {isEditingPrompt ? (
                    <textarea
                      value={promptText}
                      onChange={(e) => {
                        setPromptText(e.target.value);
                        e.target.style.height = 'auto';
                        e.target.style.height = Math.max(60, e.target.scrollHeight) + 'px';
                      }}
                      onBlur={async () => {
                        if (promptText !== systemPrompt) {
                          await updateSystemPrompt();
                        } else {
                          setIsEditingPrompt(false);
                        }
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Escape') {
                          setPromptText(systemPrompt);
                          setIsEditingPrompt(false);
                        }
                      }}
                      onFocus={(e) => {
                        e.target.style.height = 'auto';
                        e.target.style.height = Math.max(60, e.target.scrollHeight) + 'px';
                      }}
                      className="w-full min-h-15 px-2 py-2 bg-gray-700/30 border border-gray-500/50 rounded text-gray-300 text-xs resize-y focus:outline-none focus:ring-1 focus:ring-gray-400 focus:bg-gray-700/40"
                      placeholder="Enter custom prompt..."
                      style={{ overflow: 'hidden' }}
                      autoFocus
                    />
                  ) : (
                    <div 
                      className="text-xs text-gray-300 bg-gray-700/30 rounded p-2 min-h-15 flex items-start border border-gray-600/30 cursor-text hover:bg-gray-700/40 hover:border-gray-500/40 transition-colors whitespace-pre-wrap"
                      onClick={() => {
                        setPromptText(systemPrompt);
                        setIsEditingPrompt(true);
                      }}
                      title="Click to edit prompt"
                    >
                      {systemPrompt || 'Click to set detection prompt...'}
                    </div>
                  )}
                </div>
              </div>

              {/* Performance Stats */}
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                <h2 className="text-lg font-semibold text-white mb-3">Performance Stats</h2>
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-2 bg-gray-700/30 rounded">
                    <div className="text-xl font-bold text-blue-400">{detectionResults.length}</div>
                    <div className="text-xs text-gray-400">Detections</div>
                  </div>
                  <div className="text-center p-2 bg-gray-700/30 rounded">
                    <div className="text-xl font-bold text-green-400">
                      {averageLatency > 0 ? `${averageLatency.toFixed(0)}ms` : '-'}
                    </div>
                    <div className="text-xs text-gray-400">Avg Latency</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Panel - Video Simulation & Results */}
            <div className="xl:col-span-2 space-y-6">
              {/* Video/Camera Feed Area */}
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
                <h2 className="text-xl font-semibold text-white mb-4">
                  {detectionMode === 'video' ? 'Video Feed' : 'Camera Feed'}
                </h2>
                <div className="aspect-video bg-gray-900/50 rounded-lg border border-gray-700/50 flex items-center justify-center overflow-hidden">
                  {detectionMode === 'video' ? (
                    // Video mode content
                    selectedVideo && selectedVideo.url ? (
                      <video
                        key={selectedVideo.id}
                        controls
                        className="w-full h-full object-contain"
                        style={{ maxHeight: '100%', maxWidth: '100%' }}
                      >
                        <source src={selectedVideo.url} type="video/mp4" />
                        Your browser does not support video playback.
                      </video>
                    ) : selectedVideo ? (
                      <div className="text-center">
                        <div className="text-6xl mb-4">{selectedVideo.thumbnail}</div>
                        <div className="text-xl font-medium text-white mb-2">{selectedVideo.name}</div>
                        <div className="text-gray-400">{selectedVideo.description}</div>
                        <div className="text-sm text-yellow-400 mt-2">Video file not available for streaming</div>
                        {isDetecting && (
                          <div className="mt-4 flex items-center justify-center space-x-2">
                            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                            <span className="text-red-400 font-medium">DETECTING</span>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center text-gray-500">
                        <div className="text-4xl mb-2">📹</div>
                        <div>Select a scenario to begin</div>
                      </div>
                    )
                  ) : (
                    // Camera mode content
                    <div className="w-full h-full relative">
                      <CameraFeed className="w-full h-full" />
                      {isDetecting && (
                        <div className="absolute top-4 left-4 flex items-center space-x-2 bg-red-600/80 text-white px-3 py-1 rounded-lg">
                          <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                          <span className="text-sm font-medium">DETECTING</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Detection Results Table */}
              <div className="bg-gray-800/50 rounded-xl border border-gray-700/50 overflow-hidden">
                <div className="p-6 border-b border-gray-700/50">
                  <h2 className="text-xl font-semibold text-white">Detection Results</h2>
                </div>
                <div className="overflow-x-auto">
                  {detectionResults.length > 0 ? (
                    <table className="w-full">
                      <thead className="bg-gray-700/30">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Timestamp</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Detected Objects</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Latency</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Frame</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Confidence</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-700/50">
                        {detectionResults.map((result) => (
                          <tr key={result.id} className="hover:bg-gray-700/20">
                            <td className="px-4 py-3 text-sm text-gray-300 font-mono">
                              {result.timestamp}
                            </td>
                            <td className="px-4 py-3 text-sm font-medium text-white">
                              {result.busNumber}
                            </td>
                            <td className="px-4 py-3 text-sm text-gray-300">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                result.latency < 1000 ? 'bg-green-600/20 text-green-400' :
                                result.latency < 2000 ? 'bg-yellow-600/20 text-yellow-400' :
                                'bg-red-600/20 text-red-400'
                              }`}>
                                {result.latency}ms
                              </span>
                            </td>
                            <td className="px-4 py-3 text-sm text-gray-300">
                              {result.frameUrl ? (
                                <div 
                                  className="w-20 h-12 border border-gray-600 rounded overflow-hidden cursor-pointer hover:border-blue-500 transition-colors"
                                  onClick={() => setExpandedFrame(result)}
                                >
                                  <img 
                                    src={result.frameUrl} 
                                    alt={`Frame ${result.frameIndex}`}
                                    className="w-full h-full object-cover hover:scale-105 transition-transform"
                                    onError={(e) => {
                                      e.currentTarget.style.display = 'none';
                                      e.currentTarget.nextElementSibling?.classList.remove('hidden');
                                    }}
                                  />
                                  <div className="hidden w-full h-full bg-gray-700 flex items-center justify-center text-xs">
                                    Frame {result.frameIndex}
                                  </div>
                                </div>
                              ) : (
                                <div className="w-20 h-12 bg-gray-700 rounded border border-gray-600 flex items-center justify-center text-xs">
                                  Frame {result.frameIndex}
                                </div>
                              )}
                            </td>
                            <td className="px-4 py-3 text-sm text-gray-300">
                              {result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="p-8 text-center text-gray-500">
                      <div className="text-4xl mb-4">🔍</div>
                      <div className="text-lg mb-2">No detections yet</div>
                      <div className="text-sm">Start detection to see objects identified by Moondream</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp4,.mov"
        onChange={handleFileUpload}
        className="hidden"
      />

      {/* Expanded Frame Modal */}
      {expandedFrame && (
        <div 
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4"
          onClick={() => setExpandedFrame(null)}
        >
          <div 
            className="bg-gray-800 rounded-xl p-6 max-w-4xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-xl font-semibold text-white mb-2">Detection Frame Details</h3>
                <div className="text-sm text-gray-400 space-y-1">
                  <div>Video Time: <span className="text-white font-mono">{expandedFrame.timestamp}</span></div>
                  <div>Bus Number: <span className="text-yellow-400 font-semibold">{expandedFrame.busNumber}</span></div>
                  <div>Latency: <span className="text-green-400">{expandedFrame.latency}ms</span></div>
                  <div>Frame Index: <span className="text-blue-400">{expandedFrame.frameIndex}</span></div>
                </div>
              </div>
              <button
                onClick={() => setExpandedFrame(null)}
                className="text-gray-400 hover:text-white text-2xl leading-none"
              >
                ✕
              </button>
            </div>
            
            <div className="flex justify-center">
              {expandedFrame.frameUrl ? (
                <img 
                  src={expandedFrame.frameUrl} 
                  alt={`Expanded Frame ${expandedFrame.frameIndex}`}
                  className="max-w-full max-h-[60vh] object-contain rounded-lg border border-gray-600"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                    e.currentTarget.nextElementSibling?.classList.remove('hidden');
                  }}
                />
              ) : (
                <div className="w-96 h-64 bg-gray-700 rounded-lg border border-gray-600 flex items-center justify-center text-gray-400">
                  No frame image available
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}