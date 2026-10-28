import { useEffect, useState } from 'react'
import { WifiOff } from 'lucide-react'
export default function OfflineBanner() {
  const [online, setOnline] = useState(navigator.onLine)
  useEffect(() => {
    const goOnline  = () => setOnline(true)
    const goOffline = () => setOnline(false)
    window.addEventListener('online', goOnline)
    window.addEventListener('offline', goOffline)
    return () => {
      window.removeEventListener('online', goOnline)
      window.removeEventListener('offline', goOffline)
    }
  }, [])
  if (online) return null
  return (
    <div className="sticky top-14 z-40 bg-red-600 text-white text-xs font-medium px-4 py-2 flex items-center justify-center gap-2">
      <WifiOff className="w-3.5 h-3.5" />
      You are offline — forecast jobs and searches will fail until your connection is restored.
    </div>
  )
}
