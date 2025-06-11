'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/lib/WebSocketProvider';

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
  
  const { lastEvent } = useWebSocket();

  // Fetch available videos and status from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch videos
        const videosResponse = await fetch('http://localhost:8000/bus-demo/videos');
        if (videosResponse.ok) {
          const videosData = await videosResponse.json();
          const videos = videosData.videos || [];
          
          // Add video URLs for each video
          const videosWithUrls = await Promise.all(
            videos.map(async (video: BusVideo) => {
              try {
                const videoInfoResponse = await fetch(`http://localhost:8000/bus-demo/video/${video.id}`);
                if (videoInfoResponse.ok) {
                  return {
                    ...video,
                    url: `http://localhost:8000/bus-demo/videos/bus_video_${video.id}.mp4`
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
        const statusResponse = await fetch('http://localhost:8000/bus-demo/status');
        if (statusResponse.ok) {
          const statusData = await statusResponse.json();
          setCloudAvailable(statusData.cloud_available || false);
          setLocalAvailable(statusData.local_available || false);
          setUseCloud(statusData.current_mode === 'cloud');
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

    if (lastEvent.type === 'bus_demo') {
      console.log(`🎯 Processing bus_demo event: ${lastEvent.action}`);
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
            busNumber: string;
            frameIndex: number;
          };
          const newResult: DetectionResult = {
            id: `${Date.now()}-${Math.random()}`,
            timestamp: result.timestamp || '',
            videoTimestamp: result.videoTimestamp,
            latency: result.latency || 0,
            frameUrl: result.frameUrl || '',
            busNumber: result.busNumber || '',
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

  const startDetection = async () => {
    if (!selectedVideo) {
      console.warn('❌ No video selected for detection');
      return;
    }
    
    console.log(`🎬 Starting detection for video: ${selectedVideo.id} (${selectedVideo.name})`);
    
    try {
      const response = await fetch(`http://localhost:8000/bus-demo/start/${selectedVideo.id}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('✅ Detection started:', result.message);
        // State will be updated via WebSocket events
      } else {
        const error = await response.json();
        console.error('❌ Failed to start detection:', error.detail);
        alert(`Failed to start detection: ${error.detail}`);
      }
    } catch (error) {
      console.error('❌ Error starting detection:', error);
      alert('Error starting detection. Make sure the backend is running.');
    }
  };

  const stopDetection = async () => {
    try {
      const response = await fetch('http://localhost:8000/bus-demo/stop', {
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
      const response = await fetch('http://localhost:8000/bus-demo/inference-mode', {
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

  const averageLatency = frameCount > 0 ? totalLatency / frameCount : 0;

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden bg-gradient-to-br">
      <div className="p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="text-6xl mb-4">🚌</div>
            <h1 className="text-4xl font-bold text-white mb-2">Bus Number Detection Demo</h1>
            <p className="text-gray-400 text-lg">
              Simulate bus detection using Moondream VLM with latency measurements
            </p>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Left Panel - Video Selection & Controls */}
            <div className="xl:col-span-1 space-y-6">
              {/* Video Selection */}
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
                <h2 className="text-xl font-semibold text-white mb-4">Select Bus Scenario</h2>
                <div className="space-y-3">
                  {busVideos.length > 0 ? (
                    busVideos.map((video) => (
                      <button
                        key={video.id}
                        onClick={() => selectVideo(video)}
                        className={`w-full p-4 rounded-lg border transition-all duration-200 text-left ${
                          selectedVideo?.id === video.id
                            ? 'bg-blue-600/30 border-blue-500/50 text-blue-200'
                            : 'bg-gray-700/30 border-gray-600/50 text-gray-300 hover:bg-gray-600/30'
                        }`}
                      >
                        <div className="flex items-center space-x-3">
                          <span className="text-2xl">{video.thumbnail}</span>
                          <div className="flex-1">
                            <div className="font-medium">{video.name}</div>
                            <div className="text-sm opacity-75">{video.description}</div>
                            <div className="text-xs opacity-60">{video.duration}s duration</div>
                          </div>
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="text-center py-6 text-gray-500">
                      <div className="text-4xl mb-2">📹</div>
                      <div className="text-sm">No videos found</div>
                      <div className="text-xs mt-1">
                        Place bus_video_1.mp4, bus_video_2.mp4, etc. in backend/bus_videos/
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Inference Mode Toggle */}
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
                <h2 className="text-xl font-semibold text-white mb-4">Inference Mode</h2>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-gray-300">
                        {useCloud ? '☁️ Cloud' : '🖥️ Local'}
                      </span>
                      <span className="text-xs text-gray-500">
                        {useCloud ? 'Moondream Cloud' : 'Moondream Server'}
                      </span>
                    </div>
                    <button
                      onClick={toggleInferenceMode}
                      disabled={(!cloudAvailable && !localAvailable) || isDetecting}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                        useCloud ? 'bg-blue-600' : 'bg-gray-600'
                      } ${(!cloudAvailable && !localAvailable) || isDetecting ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          useCloud ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                  <div className="text-xs text-gray-500">
                    Cloud: {cloudAvailable ? '✅ Available' : '❌ Unavailable'} • 
                    Local: {localAvailable ? '✅ Available' : '❌ Unavailable'}
                  </div>
                </div>
              </div>

              {/* Detection Controls */}
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
                <h2 className="text-xl font-semibold text-white mb-4">Detection Controls</h2>
                <div className="space-y-4">
                  <button
                    onClick={startDetection}
                    disabled={!selectedVideo || isDetecting}
                    className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 ${
                      !selectedVideo || isDetecting
                        ? 'bg-gray-700/50 text-gray-500 cursor-not-allowed'
                        : 'bg-green-600 hover:bg-green-700 text-white shadow-lg hover:shadow-green-600/25'
                    }`}
                  >
                    {isDetecting ? (
                      <div className="flex items-center justify-center space-x-2">
                        <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                        <span>Detecting...</span>
                      </div>
                    ) : (
                      '🔍 Watch for Bus Numbers'
                    )}
                  </button>
                  
                  {isDetecting && (
                    <button
                      onClick={stopDetection}
                      className="w-full py-2 px-4 rounded-lg font-medium bg-red-600 hover:bg-red-700 text-white transition-all duration-200"
                    >
                      Stop Detection
                    </button>
                  )}
                </div>
              </div>

              {/* Stats */}
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
                <h2 className="text-xl font-semibold text-white mb-4">Performance Stats</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-gray-700/30 rounded-lg">
                    <div className="text-2xl font-bold text-blue-400">{detectionResults.length}</div>
                    <div className="text-sm text-gray-400">Detections</div>
                  </div>
                  <div className="text-center p-3 bg-gray-700/30 rounded-lg">
                    <div className="text-2xl font-bold text-green-400">
                      {averageLatency > 0 ? `${averageLatency.toFixed(0)}ms` : '-'}
                    </div>
                    <div className="text-sm text-gray-400">Avg Latency</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Panel - Video Simulation & Results */}
            <div className="xl:col-span-2 space-y-6">
              {/* Video Simulation Area */}
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/50">
                <h2 className="text-xl font-semibold text-white mb-4">Realtime Video Feed</h2>
                <div className="aspect-video bg-gray-900/50 rounded-lg border border-gray-700/50 flex items-center justify-center overflow-hidden">
                  {selectedVideo && selectedVideo.url ? (
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
                      <div>Select a bus scenario to begin</div>
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
                          <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Bus Number</th>
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
                      <div className="text-sm">Start detection to see bus numbers identified by Moondream</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

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