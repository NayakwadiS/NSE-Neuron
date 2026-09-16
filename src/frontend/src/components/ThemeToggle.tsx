import type { Theme } from '../hooks/useTheme'

interface Props {
  theme: Theme
  onToggle: () => void
}

export default function ThemeToggle({ theme, onToggle }: Props) {
  const isDark = theme === 'dark'
  return (
    <button
      onClick={onToggle}
      aria-label="Toggle light/dark theme"
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="ml-4 flex items-center justify-center w-8 h-8 rounded-lg border
                 border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800
                 text-slate-600 dark:text-slate-300 hover:border-indigo-400
                 dark:hover:border-indigo-500 hover:text-indigo-500 transition"
    >
      {isDark ? (
        <span className="text-sm">☀️</span>
      ) : (
        <span className="text-sm">🌙</span>
      )}
    </button>
  )
}

