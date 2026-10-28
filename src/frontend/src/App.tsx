import { useState } from 'react'
import Navbar, { type Page } from './components/Navbar'
import Home    from './pages/Home'
import CacheDashboard from './pages/CacheDashboard'
import { Toaster } from 'react-hot-toast'
import { useTheme } from './hooks/useTheme'
import OfflineBanner from './components/OfflineBanner'
import ErrorBoundary  from './components/ErrorBoundary'

export default function App() {
  const { theme, toggleTheme } = useTheme()
  const [page, setPage] = useState<Page>('home')

  return (
    <div className="min-h-screen bg-slate-100 dark:bg-slate-950 font-sans transition-colors">
      <Toaster
        position="top-right"
        toastOptions={{
          style: theme === 'dark'
            ? { background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155' }
            : { background: '#ffffff', color: '#0f172a', border: '1px solid #e2e8f0' },
        }}
      />
      <Navbar theme={theme} onToggleTheme={toggleTheme} page={page} onNavigate={setPage} />
      <OfflineBanner />
      <ErrorBoundary label="The application hit an unexpected error. Please refresh the page.">
        {page === 'home' ? <Home /> : <CacheDashboard />}
      </ErrorBoundary>
    </div>
  )
}





