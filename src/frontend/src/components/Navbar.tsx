import ThemeToggle from './ThemeToggle'
import type { Theme } from '../hooks/useTheme'

interface Props {
  theme: Theme
  onToggleTheme: () => void
}

export default function Navbar({ theme, onToggleTheme }: Props) {
  return (
    <header className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-screen-xl mx-auto px-4 h-14 flex items-center gap-3">
        {/* logo */}
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
            N
          </div>
          <span className="font-semibold text-slate-900 dark:text-white text-base tracking-tight">
            NSE<span className="text-indigo-500 dark:text-indigo-400">-Neuron</span>
          </span>
        </div>
        <span className="ml-auto text-xs text-slate-500 dark:text-slate-500 font-mono">
          AI-Powered NSE Forecasting
        </span>
        <ThemeToggle theme={theme} onToggle={onToggleTheme} />
      </div>
    </header>
  )
}



