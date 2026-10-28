import { useEffect, useMemo, useState } from 'react'
import toast from 'react-hot-toast'
import { Trash2, RefreshCw, Database, Search, ArrowUpDown, XSquare } from 'lucide-react'
import {
  fetchCachedModels,
  deleteCachedModel,
  deleteCachedModels,
  purgeExpiredModels,
} from '../api/client'
import type { CachedModel } from '../types'
import ConfirmDialog from '../components/ConfirmDialog'
import LoadingSpinner from '../components/LoadingSpinner'

type SortKey = 'symbol' | 'model' | 'trained_at' | 'n_rows' | 'size_kb'

export default function CacheDashboard() {
  const [loading, setLoading] = useState(true)
  const [models, setModels] = useState<CachedModel[]>([])
  const [meta, setMeta] = useState<{ enabled: boolean; max_stale_days: number; max_age_days: number } | null>(null)
  const [query, setQuery] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('trained_at')
  const [sortDesc, setSortDesc] = useState(true)
  const [purging, setPurging] = useState(false)
  const [confirmPurge, setConfirmPurge] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<CachedModel | null>(null)
  const [pendingDeleteAll, setPendingDeleteAll] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const res = await fetchCachedModels()
      setModels(res.models)
      setMeta({ enabled: res.enabled, max_stale_days: res.max_stale_days, max_age_days: res.max_age_days })
    } catch {
      toast.error('Failed to load cached models')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) setSortDesc(d => !d)
    else { setSortKey(key); setSortDesc(true) }
  }

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase()
    let rows = q
      ? models.filter(m => m.symbol.includes(q) || m.model.toUpperCase().includes(q))
      : models
    rows = [...rows].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey]
      const cmp = typeof av === 'string' ? String(av).localeCompare(String(bv)) : (Number(av) - Number(bv))
      return sortDesc ? -cmp : cmp
    })
    return rows
  }, [models, query, sortKey, sortDesc])

  const totalSizeKb = useMemo(() => models.reduce((s, m) => s + m.size_kb, 0), [models])

  const handleDeleteOne = async () => {
    if (!pendingDelete) return
    const { symbol, model } = pendingDelete
    setPendingDelete(null)
    try {
      await deleteCachedModel(symbol, model)
      toast.success(`Removed ${symbol}/${model}`)
      load()
    } catch {
      toast.error('Failed to delete')
    }
  }

  const handleDeleteAllForSymbol = async () => {
    if (!pendingDeleteAll) return
    const symbol = pendingDeleteAll
    setPendingDeleteAll(null)
    try {
      const res = await deleteCachedModels(symbol)
      toast.success(`Cleared ${res.deleted} cached model(s) for ${symbol}`)
      load()
    } catch {
      toast.error('Failed to clear cache')
    }
  }

  const handlePurge = async () => {
    setConfirmPurge(false)
    setPurging(true)
    try {
      const res = await purgeExpiredModels()
      toast.success(res.purged > 0 ? `Purged ${res.purged} expired model(s)` : 'No expired models to purge')
      load()
    } catch {
      toast.error('Failed to purge expired models')
    } finally {
      setPurging(false)
    }
  }

  const SortHeader = ({ label, k, className = '' }: { label: string; k: SortKey; className?: string }) => (
    <th
      onClick={() => toggleSort(k)}
      className={`px-3 py-2 text-left font-semibold text-slate-500 dark:text-slate-400 cursor-pointer select-none hover:text-slate-800 dark:hover:text-slate-200 ${className}`}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <ArrowUpDown className={`w-3 h-3 ${sortKey === k ? 'opacity-100' : 'opacity-30'}`} />
      </span>
    </th>
  )

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-6">
      <ConfirmDialog
        open={confirmPurge}
        title="Purge expired models?"
        message={`This will permanently delete every cached model older than ${meta?.max_age_days ?? '?'} days.`}
        confirmLabel="Purge expired"
        danger
        onConfirm={handlePurge}
        onCancel={() => setConfirmPurge(false)}
      />
      <ConfirmDialog
        open={!!pendingDelete}
        title="Delete cached model?"
        message={pendingDelete ? `Permanently delete the cached ${pendingDelete.model.toUpperCase()} weights for ${pendingDelete.symbol}?` : ''}
        confirmLabel="Delete"
        danger
        onConfirm={handleDeleteOne}
        onCancel={() => setPendingDelete(null)}
      />
      <ConfirmDialog
        open={!!pendingDeleteAll}
        title="Clear all cached models?"
        message={pendingDeleteAll ? `Permanently delete every cached model for ${pendingDeleteAll}?` : ''}
        confirmLabel="Clear all"
        danger
        onConfirm={handleDeleteAllForSymbol}
        onCancel={() => setPendingDeleteAll(null)}
      />

      <div className="flex items-center justify-between gap-4 flex-wrap mb-5">
        <div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-indigo-500" />
            Cached Models
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            {models.length} model{models.length !== 1 ? 's' : ''} cached · {(totalSizeKb / 1024).toFixed(2)} MB total
            {meta && !meta.enabled && <span className="text-amber-500 ml-2">(caching currently disabled)</span>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={load}
            className="px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-xs font-medium text-slate-600 dark:text-slate-300 hover:border-slate-400 dark:hover:border-slate-600 flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
          <button
            onClick={() => setConfirmPurge(true)}
            disabled={purging || models.length === 0}
            className="px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 text-xs font-medium text-red-500 hover:border-red-500/40 disabled:opacity-40 flex items-center gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" /> {purging ? 'Purging…' : 'Purge expired'}
          </button>
        </div>
      </div>

      <div className="relative mb-4 max-w-sm">
        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Filter by symbol or model…"
          className="w-full pl-9 pr-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
      </div>

      {loading && <LoadingSpinner message="Loading cached models…" />}

      {!loading && filtered.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center text-slate-400 dark:text-slate-600">
          <Database className="w-10 h-10 mb-3 text-slate-300 dark:text-slate-700" />
          <p className="text-sm">
            {models.length === 0 ? 'No models cached yet. Run a forecast to populate this list.' : 'No models match your filter.'}
          </p>
        </div>
      )}

      {!loading && filtered.length > 0 && (
        <div className="overflow-auto rounded-xl border border-slate-300 dark:border-slate-800 max-h-[560px]">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-slate-100 dark:bg-slate-800 z-10">
              <tr>
                <SortHeader label="Symbol" k="symbol" />
                <SortHeader label="Model" k="model" />
                <SortHeader label="Trained At" k="trained_at" />
                <SortHeader label="Rows" k="n_rows" />
                <SortHeader label="Size" k="size_kb" />
                <th className="px-3 py-2 text-left font-semibold text-slate-500 dark:text-slate-400">Last Date</th>
                <th className="px-3 py-2 text-right font-semibold text-slate-500 dark:text-slate-400">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {filtered.map(m => (
                <tr key={`${m.symbol}-${m.model}`} className="bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800/50">
                  <td className="px-3 py-2 font-mono font-semibold text-slate-800 dark:text-slate-200">{m.symbol}</td>
                  <td className="px-3 py-2 uppercase text-indigo-600 dark:text-indigo-400 font-medium">{m.model}</td>
                  <td className="px-3 py-2 text-slate-600 dark:text-slate-400">{new Date(m.trained_at).toLocaleString()}</td>
                  <td className="px-3 py-2 text-slate-600 dark:text-slate-400">{m.n_rows}</td>
                  <td className="px-3 py-2 text-slate-600 dark:text-slate-400">{m.size_kb.toFixed(1)} KB</td>
                  <td className="px-3 py-2 text-slate-600 dark:text-slate-400">{m.last_date}</td>
                  <td className="px-3 py-2 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <button
                        onClick={() => setPendingDeleteAll(m.symbol)}
                        title={`Clear all cached models for ${m.symbol}`}
                        className="p-1.5 rounded-md hover:bg-amber-500/10 text-slate-400 hover:text-amber-500 transition"
                      >
                        <XSquare className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => setPendingDelete(m)}
                        title="Delete this cached model"
                        className="p-1.5 rounded-md hover:bg-red-500/10 text-slate-400 hover:text-red-500 transition"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}



