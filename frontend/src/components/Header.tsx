'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function Header() {
  const pathname = usePathname();

  const tabs = [
    { href: '/', label: 'Osmo', icon: '🤖', subtitle: 'AI Assistant' },
    { href: '/object-demo', label: 'Bus Demo', icon: '🚌', subtitle: 'Transportation' },
  ];

  return (
    <header className="flex-shrink-0 p-6">
      <div className="max-w-8xl mx-auto">
        <div className="flex items-center justify-center">
          <div className="flex items-center space-x-3 bg-gray-800/50 backdrop-blur-sm rounded-xl p-2 border border-gray-700/50 shadow-2xl">
            {tabs.map((tab) => {
              const isActive = pathname === tab.href;
              const isBusDemo = tab.href === '/object-demo';
              
              // Different gradients for different tabs - less intense yellow
              const activeGradient = isBusDemo 
                ? 'bg-gradient-to-r from-yellow-700/80 to-amber-700/80 text-white shadow-lg ' 
                : 'bg-gradient-to-r from-gray-700 to-gray-600 text-white shadow-lg ';
              
              const activeOverlay = isBusDemo
                ? 'bg-gradient-to-r from-yellow-500/5 to-amber-500/5'
                : 'bg-gradient-to-r from-blue-500/10 to-purple-500/10';

              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  className={`
                    relative flex items-center px-6 py-3 rounded-lg font-medium transition-all duration-300 min-w-[160px]
                    ${isActive
                      ? `${activeGradient} transform scale-105`
                      : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/30 filter grayscale hover:grayscale-0'
                    }
                  `}
                >
                  <div className={`text-2xl mr-4 transition-all duration-300 ${isActive ? 'scale-110' : 'scale-90 opacity-60'}`}>
                    {tab.icon}
                  </div>
                  <div className="flex-1">
                    <div className={`text-sm font-bold tracking-tight ${isActive ? 'text-white' : 'text-gray-500'}`}>
                      {tab.label}
                    </div>
                    <div className={`text-xs ${isActive ? 'text-gray-100' : 'text-gray-600'}`}>
                      {tab.subtitle}
                    </div>
                  </div>
                  {isActive && (
                    <div className={`absolute inset-0 rounded-lg ${activeOverlay} pointer-events-none`} />
                  )}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </header>
  );
} 