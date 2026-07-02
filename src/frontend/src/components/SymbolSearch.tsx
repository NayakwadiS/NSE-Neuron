import { useState, useRef, useEffect } from 'react'
import { searchSymbols } from '../api/client'
import type { SymbolResult } from '../types'

interface Props {
  value:    string
  onChange: (symbol: string, name: string) => void
}

export default function SymbolSearch({ value, onChange }: Props) {
  const [query,    setQuery]    = useState(value)
  const [results,  setResults]  = useState<SymbolResult[]>([])
  const [open,     setOpen]     = useState(false)
  const [loading,  setLoading]  = useState(false)
  const timerRef                 = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wrapRef                  = useRef<HTMLDivElement>(null)

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleInput = (v: string) => {
    setQuery(v)
    if (timerRef.current) clearTimeout(timerRef.current)
    if (v.trim().length < 1) { setResults([]); setOpen(false); return }
    timerRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const res = await searchSymbols(v)
        setResults(res)
        setOpen(true)
      } catch { /* ignore */ } finally { setLoading(false) }
    }, 350)
  }

  const pick = (sym: SymbolResult) => {
    setQuery(sym.symbol)
    setOpen(false)
    onChange(sym.symbol, sym.name)
  }

  return (
    <div ref={wrapRef} className="relative">
      <label className="block text-xs font-medium text-slate-400 mb-1.5">
        NSE Symbol
      </label>
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={e => handleInput(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder="e.g. INFY, TCS, SBIN…"
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2.5
                     text-sm text-white placeholder-slate-500 outline-none
                     focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/40 transition"
        />
        {loading && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 animate-spin text-base">
            ⟳
          </span>
        )}
      </div>

      {open && results.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 max-h-56 overflow-y-auto
                        bg-slate-800 border border-slate-700 rounded-lg shadow-xl">
          {results.map(sym => (
            <li
              key={sym.symbol}
              onMouseDown={() => pick(sym)}
              className="px-3 py-2 cursor-pointer hover:bg-slate-700 flex items-baseline gap-2"
            >
              <span className="font-mono text-sm font-semibold text-indigo-400">{sym.symbol}</span>
              <span className="text-xs text-slate-400 truncate">{sym.name}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

