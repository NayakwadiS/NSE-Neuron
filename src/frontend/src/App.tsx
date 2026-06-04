import Navbar  from './components/Navbar'
import Home    from './pages/Home'
import { Toaster } from 'react-hot-toast'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 font-sans">
      <Toaster
        position="top-right"
        toastOptions={{
          style: { background: '#1e293b', color: '#f1f5f9', border: '1px solid #334155' },
        }}
      />
      <Navbar />
      <Home />
    </div>
  )
}

