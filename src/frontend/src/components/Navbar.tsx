import ThemeToggle from './ThemeToggle'
import type { Theme } from '../hooks/useTheme'
import { Database, LineChart } from 'lucide-react'

export type Page = 'home' | 'cache'

interface Props {
  theme: Theme
  onToggleTheme: () => void
  page: Page
  onNavigate: (page: Page) => void
}

export default function Navbar({ theme, onToggleTheme, page, onNavigate }: Props) {
  return (
    <header className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-screen-xl mx-auto px-4 h-14 flex items-center gap-3">
        {/* logo */}
        <button onClick={() => onNavigate('home')} className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
            N
          </div>
          <span className="font-semibold text-slate-900 dark:text-white text-base tracking-tight">
            NSE<span className="text-indigo-500 dark:text-indigo-400">-Neuron</span>
          </span>
        </button>

        <nav className="flex items-center gap-1 ml-2">
          <button
            onClick={() => onNavigate('home')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-1.5 transition ${
              page === 'home'
                ? 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400'
                : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <LineChart className="w-3.5 h-3.5" /> Forecast
          </button>
          <button
            onClick={() => onNavigate('cache')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-1.5 transition ${
              page === 'cache'
                ? 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400'
                : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" /> Cached Models
          </button>
        </nav>

        <span className="ml-auto text-xs text-slate-500 dark:text-slate-500 font-mono hidden sm:inline">
          AI-Powered NSE Forecasting
        </span>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  )
}



