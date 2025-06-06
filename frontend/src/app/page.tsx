'use client';

import { useSocket } from '@/lib/useSocket';
import { CameraFeed } from '@/components/CameraFeed';
import { ChatPanel } from '@/components/ChatPanel';
import { StatusBar } from '@/components/StatusBar';
import { DebugStream } from '@/components/DebugStream';
import { ToolIndicator } from '@/components/ToolIndicator';

export default function Home() {
  const { isConnected, lastEvent, systemStatus, toolState, error } = useSocket('ws://localhost:8000/ws');

  return (
    <div className="h-screen bg-gray-950 flex flex-col overflow-hidden">
      {/* Header - Fixed height */}
      <header className="flex-shrink-0 p-3 sm:p-4 pb-3">
        <div className="flex items-center space-x-2 sm:space-x-3">
          <div className="text-2xl sm:text-3xl">🤖</div>
          <div className="flex-1 min-w-0">
            <h1 className="text-xl sm:text-2xl font-bold text-white truncate">Osmo Assistant</h1>
            <p className="text-gray-400 text-xs sm:text-sm">Always-listening, always-watching AI</p>
          </div>
          <div className="flex-shrink-0">
            <div className={`flex items-center space-x-2 px-2 sm:px-3 py-1 rounded-full text-xs ${
              isConnected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-400' : 'bg-red-400'
              }`} />
              <span className="hidden sm:inline">{isConnected ? 'Connected' : 'Disconnected'}</span>
              <span className="sm:hidden">{isConnected ? '✓' : '✗'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - Responsive layout */}
      <main className="flex-1 flex flex-col lg:flex-row gap-3 sm:gap-4 px-3 sm:px-4 pb-3 sm:pb-4 min-h-0 overflow-hidden">
        {/* Left Panel - Camera and Status/Debug */}
        <div className="flex flex-col gap-3 sm:gap-4 lg:w-3/5 xl:w-3/5 min-h-0">
          {/* Camera Feed - 16:9 aspect ratio scaled down */}
          <div className="flex-shrink-0 relative w-full max-w-2xl mx-auto" style={{ aspectRatio: '16/9' }}>
            <CameraFeed className="h-full w-full" />
            
            {/* Tool indicator overlay */}
            <div className="absolute top-2 sm:top-4 left-2 sm:left-4 z-10">
              <ToolIndicator toolState={toolState} />
            </div>
          </div>
          
          {/* Status and Debug - More space */}
          <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 min-h-0">
            <StatusBar
              isConnected={isConnected}
              systemStatus={systemStatus}
              lastEvent={lastEvent}
              error={error}
              className="h-full min-h-[200px]"
            />
            <DebugStream
              lastEvent={lastEvent}
              className="h-full min-h-[200px]"
            />
          </div>
        </div>

        {/* Right Panel - Chat Panel with full height */}
        <div className="flex-1 lg:w-2/5 xl:w-2/5 min-h-0">
          <ChatPanel 
            lastEvent={lastEvent} 
            className="h-full"
          />
        </div>
      </main>
    </div>
  );
}
