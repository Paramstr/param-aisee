'use client';

import { useSocket } from '@/lib/useSocket';
import { CameraFeed } from '@/components/CameraFeed';
import { ChatPanel } from '@/components/ChatPanel';
import { StatusBar } from '@/components/StatusBar';

export default function Home() {
  const { isConnected, lastEvent, systemStatus, error } = useSocket('ws://localhost:8000/ws');

  return (
    <div className="min-h-screen bg-gray-950 p-4">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center space-x-3">
          <div className="text-3xl">🤖</div>
          <div>
            <h1 className="text-2xl font-bold text-white">Osmo Assistant</h1>
            <p className="text-gray-400 text-sm">Always-listening, always-watching AI</p>
          </div>
          <div className="ml-auto">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs ${
              isConnected ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-400' : 'bg-red-400'
              }`} />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-140px)]">
        {/* Camera Feed - Takes up 2 columns on large screens */}
        <div className="lg:col-span-2">
          <CameraFeed className="h-full min-h-[400px]" />
        </div>

        {/* Right Panel - Chat and Status with proper flex layout */}
        <div className="flex flex-col gap-4 h-full">
          {/* Chat Panel - Takes up most space */}
          <div className="flex-1 min-h-0">
            <ChatPanel 
              lastEvent={lastEvent} 
              className="h-full"
            />
          </div>
          
          {/* Status Bar - Fixed height */}
          <div className="flex-shrink-0">
            <StatusBar
              isConnected={isConnected}
              systemStatus={systemStatus}
              lastEvent={lastEvent}
              error={error}
              className="h-full"
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-6 text-center text-xs text-gray-500">
        <p>Say &quot;Osmo&quot; followed by your question to interact • Camera and microphone permissions required</p>
      </div>
    </div>
  );
}
