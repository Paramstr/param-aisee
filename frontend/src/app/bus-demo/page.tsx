'use client';

export default function BusDemo() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center p-8">
      <div className="max-w-md space-y-6">
        <div className="text-8xl mb-6">🚌</div>
        <h2 className="text-3xl font-bold text-white">Bus Demo</h2>
        <p className="text-gray-400 text-lg">
          This page is currently empty. Bus demo functionality will be implemented here.
        </p>
        <div className="mt-8 p-6 bg-gray-800/50 rounded-lg border border-gray-700/50">
          <h3 className="text-lg font-semibold text-white mb-3">Coming Soon</h3>
          <ul className="text-sm text-gray-400 space-y-2 text-left">
            <li>• Bus route optimization</li>
            <li>• Real-time tracking</li>
            <li>• Passenger management</li>
            <li>• Fleet analytics</li>
          </ul>
        </div>
      </div>
    </div>
  );
} 