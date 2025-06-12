'use client';

import { useState, useEffect, useRef } from 'react';

interface AudioDevice {
  id: number;
  name: string;
  channels: number;
  sample_rate: number;
  is_default: boolean;
}

interface VideoDevice {
  id: number;
  name: string;
  is_current: boolean;
}

interface DeviceSelectorProps {
  className?: string;
}

export function DeviceSelector({ className = '' }: DeviceSelectorProps) {
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([]);
  const [videoDevices, setVideoDevices] = useState<VideoDevice[]>([]);
  const [currentAudioDevice, setCurrentAudioDevice] = useState<number | null>(null);
  const [currentVideoDevice, setCurrentVideoDevice] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioDropdownOpen, setAudioDropdownOpen] = useState(false);
  const [videoDropdownOpen, setVideoDropdownOpen] = useState(false);
  
  const audioDropdownRef = useRef<HTMLDivElement>(null);
  const videoDropdownRef = useRef<HTMLDivElement>(null);

  const fetchDevices = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Fetch audio devices
      const audioResponse = await fetch('/api/audio_devices');
      if (!audioResponse.ok) throw new Error('Failed to fetch audio devices');
      const audioData = await audioResponse.json();
      setAudioDevices(audioData.devices);
      setCurrentAudioDevice(audioData.current_device);

      // Fetch video devices
      const videoResponse = await fetch('/api/video_devices');
      if (!videoResponse.ok) throw new Error('Failed to fetch video devices');
      const videoData = await videoResponse.json();
      setVideoDevices(videoData.devices);
      setCurrentVideoDevice(videoData.current_device);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch devices');
    } finally {
      setIsLoading(false);
    }
  };

  const updateDevice = async (deviceType: 'audio' | 'video', deviceId: number) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/device_update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          device_type: deviceType,
          device_id: deviceId,
        }),
      });

      if (!response.ok) throw new Error(`Failed to update ${deviceType} device`);

      // Update current device state
      if (deviceType === 'audio') {
        setCurrentAudioDevice(deviceId);
        setAudioDropdownOpen(false);
      } else {
        setCurrentVideoDevice(deviceId);
        setVideoDropdownOpen(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to update ${deviceType} device`);
    } finally {
      setIsLoading(false);
    }
  };

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (audioDropdownRef.current && !audioDropdownRef.current.contains(event.target as Node)) {
        setAudioDropdownOpen(false);
      }
      if (videoDropdownRef.current && !videoDropdownRef.current.contains(event.target as Node)) {
        setVideoDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    fetchDevices();
  }, []);

  const getCurrentAudioDeviceName = () => {
    const device = audioDevices.find(d => d.id === currentAudioDevice);
    return device ? (device.name.length > 20 ? device.name.substring(0, 20) + '...' : device.name) : 'No audio device';
  };

  const getCurrentVideoDeviceName = () => {
    const device = videoDevices.find(d => d.id === currentVideoDevice);
    return device ? (device.name.length > 20 ? device.name.substring(0, 20) + '...' : device.name) : 'No video device';
  };

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {/* Audio Device Dropdown */}
      <div className="relative" ref={audioDropdownRef}>
        <button
          onClick={() => setAudioDropdownOpen(!audioDropdownOpen)}
          disabled={isLoading}
          className="flex items-center space-x-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 text-white text-xs rounded border border-gray-600 transition-colors"
          title={`Audio: ${getCurrentAudioDeviceName()}`}
        >
          <span>🎤</span>
          <span className="hidden sm:inline">{getCurrentAudioDeviceName()}</span>
          <svg className="w-3 h-3 ml-1" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>

        {audioDropdownOpen && (
          <div className="absolute top-full left-0 mt-1 w-64 bg-gray-800 border border-gray-600 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
            <div className="p-2">
              <div className="text-xs text-gray-400 mb-2">Audio Devices</div>
              {audioDevices.map((device) => (
                <button
                  key={device.id}
                  onClick={() => updateDevice('audio', device.id)}
                  className={`w-full text-left p-2 rounded text-xs hover:bg-gray-700 transition-colors ${
                    device.id === currentAudioDevice ? 'bg-blue-900/50 text-blue-300' : 'text-gray-300'
                  }`}
                  disabled={isLoading}
                >
                  <div className="font-medium truncate">{device.name}</div>
                  <div className="text-gray-500 text-xs">
                    {device.channels} ch • {device.sample_rate.toFixed(0)} Hz
                    {device.is_default && ' (Default)'}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Video Device Dropdown */}
      <div className="relative" ref={videoDropdownRef}>
        <button
          onClick={() => setVideoDropdownOpen(!videoDropdownOpen)}
          disabled={isLoading}
          className="flex items-center space-x-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 text-white text-xs rounded border border-gray-600 transition-colors"
          title={`Video: ${getCurrentVideoDeviceName()}`}
        >
          <span>📹</span>
          <span className="hidden sm:inline">{getCurrentVideoDeviceName()}</span>
          <svg className="w-3 h-3 ml-1" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>

        {videoDropdownOpen && (
          <div className="absolute top-full left-0 mt-1 w-64 bg-gray-800 border border-gray-600 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
            <div className="p-2">
              <div className="text-xs text-gray-400 mb-2">Video Devices</div>
              {videoDevices.map((device) => (
                <button
                  key={device.id}
                  onClick={() => updateDevice('video', device.id)}
                  className={`w-full text-left p-2 rounded text-xs hover:bg-gray-700 transition-colors ${
                    device.id === currentVideoDevice ? 'bg-blue-900/50 text-blue-300' : 'text-gray-300'
                  }`}
                  disabled={isLoading}
                >
                  <div className="font-medium truncate">{device.name}</div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Error indicator */}
      {error && (
        <div className="text-red-400 text-xs" title={error}>
          ⚠️
        </div>
      )}
    </div>
  );
} 