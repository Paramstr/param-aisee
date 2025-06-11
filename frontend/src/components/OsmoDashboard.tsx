'use client';

import { useSocket } from '@/lib/useSocket';
import { CameraFeed } from '@/components/CameraFeed';
import { ChatPanel } from '@/components/ChatPanel';
import { StatusBar } from '@/components/StatusBar';
import { DebugStream } from '@/components/DebugStream';
import { ToolIndicator } from '@/components/ToolIndicator';
import { DeviceSelector } from '@/components/DeviceSelector';

export function OsmoDashboard() {
  const { isConnected, lastEvent, systemStatus, toolState, error, refreshSystemStatus } = useSocket('ws://localhost:8000/ws');

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header Controls */}
      <div className="flex-shrink-0 p-4">
        <div className="max-w-8xl mx-auto">
          <div className="rounded-xl p-3 fade-in">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="flex-1 min-w-0">
                  <h2 className="text-lg font-semibold text-white tracking-tight">Dashboard</h2>
                  <p className="text-xs text-gray-400 font-medium">Always-listening, always-watching AI</p>
                </div>
              </div>
              
              {/* Center - Tool Indicator */}
              <div className="flex-1 flex justify-center">
                <ToolIndicator toolState={toolState} />
              </div>
              
              <div className="flex items-center space-x-3">
                {/* Device Selector */}
                <DeviceSelector />
                
                {/* Connection Status */}
                <div className={`inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                  isConnected 
                    ? 'bg-green-900/50 text-green-300 border border-green-800/50' 
                    : 'bg-red-900/50 text-red-300 border border-red-800/50'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full mr-2 ${
                    isConnected ? 'bg-green-400 shadow-lg shadow-green-400/50' : 'bg-red-400 shadow-lg shadow-red-400/50'
                  }`} />
                  <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Optimized spacing */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <div className="h-full max-w-8xl mx-auto px-4 pb-4 flex flex-col">
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0">
            
            {/* Left Panel - Camera and Status */}
            <div className="lg:col-span-7 flex flex-col gap-4 min-h-0">
              
              {/* Camera Feed - Smaller aspect ratio */}
              <div className="relative flex-shrink-0">
                <div className="elevated-card rounded-xl overflow-hidden fade-in" style={{ aspectRatio: '16/10', maxHeight: '280px' }}>
                  <CameraFeed className="h-full w-full object-cover" />
                  
                  {/* Status overlay */}
                  <div className="absolute top-3 left-3 z-10 glass-card rounded-lg px-4 py-1">
                    <div className="flex items-center justify-between text-xs space-x-2">
                      <span className="text-gray-300 font-medium">Live Feed</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse"></div>
                        <span className="text-gray-400">Recording</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Status and Debug Grid - Compact height */}
              <div className="flex-shrink-0 grid grid-cols-1 md:grid-cols-2 gap-4" style={{ height: '240px' }}>
                <div className="fade-in" style={{ animationDelay: '0.1s' }}>
                  <StatusBar
                    isConnected={isConnected}
                    systemStatus={systemStatus}
                    lastEvent={lastEvent}
                    error={error}
                    refreshSystemStatus={refreshSystemStatus}
                    className="h-full"
                  />
                </div>
                
                <div className="fade-in" style={{ animationDelay: '0.2s' }}>
                  <DebugStream
                    lastEvent={lastEvent}
                    className="h-full"
                  />
                </div>
              </div>
            </div>

            {/* Right Panel - Chat */}
            <div className="lg:col-span-5 min-h-0 fade-in" style={{ animationDelay: '0.3s' }}>
              <ChatPanel 
                lastEvent={lastEvent} 
                className="h-full"
              />
            </div>
            
          </div>
        </div>
      </div>
    </div>
  );
} 