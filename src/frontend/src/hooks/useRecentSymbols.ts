import { useCallback, useEffect, useState } from 'react'
const KEY = 'nse-neuron-recent-symbols'
const MAX = 5
export interface RecentSymbol {
  symbol: string
  name:   string
}
function load(): RecentSymbol[] {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}
export function useRecentSymbols() {
  const [recents, setRecents] = useState<RecentSymbol[]>(load)
  useEffect(() => {
    localStorage.setItem(KEY, JSON.stringify(recents))
  }, [recents])
  const addRecent = useCallback((symbol: string, name: string) => {
    if (!symbol) return
    setRecents(prev => {
      const next = [{ symbol, name }, ...prev.filter(r => r.symbol !== symbol)]
      return next.slice(0, MAX)
    })
  }, [])
  return { recents, addRecent }
}
