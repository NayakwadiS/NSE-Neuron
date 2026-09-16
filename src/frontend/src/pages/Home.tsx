import { useState } from 'react'
import { startForecast, fetchRegimeAnalysis, deleteCachedModels } from '../api/client'
import { useJob } from '../hooks/useJob'
import { useRecentSymbols } from '../hooks/useRecentSymbols'
import SymbolSearch       from '../components/SymbolSearch'
import AlgorithmSelector  from '../components/AlgorithmSelector'
import LoadingSpinner     from '../components/LoadingSpinner'
import ForecastTable      from '../components/ForecastTable'
import AllAlgosTable      from '../components/AllAlgosTable'
import ForecastChart      from '../components/ForecastChart'
import RegimeCard         from '../components/RegimeCard'
import PatternList        from '../components/PatternList'
import CacheBadge         from '../components/CacheBadge'
import ErrorBoundary      from '../components/ErrorBoundary'
import ConfirmDialog      from '../components/ConfirmDialog'
import { SkeletonChart, SkeletonTable } from '../components/Skeletons'
import toast              from 'react-hot-toast'
import { Play, RotateCcw, Search, Trash2, History, LineChart } from 'lucide-react'
import type {
  SingleForecastResult,
  AllForecastResult,
  RegimeAnalysisResult,
} from '../types'

type TabId = 'forecast' | 'chart' | 'regime'

const LS_SYMBOL    = 'nse-neuron-last-symbol'
const LS_COMPANY   = 'nse-neuron-last-company'
const LS_ALGORITHM = 'nse-neuron-last-algorithm'
const LS_AUTORUN   = 'nse-neuron-autorun'

export default function Home() {
  const [symbol,    setSymbol]    = useState(() => localStorage.getItem(LS_SYMBOL) ?? '')
  const [company,   setCompany]   = useState(() => localStorage.getItem(LS_COMPANY) ?? '')
  const [algorithm, setAlgorithm] = useState(() => localStorage.getItem(LS_ALGORITHM) ?? 'lstm')
  const [jobId,     setJobId]     = useState<string | null>(null)
  const [tab,       setTab]       = useState<TabId>('forecast')
  const [regimeResult, setRegimeResult] = useState<RegimeAnalysisResult | null>(null)
  const [regimeLoading, setRegimeLoading] = useState(false)
  const [forceRetrain, setForceRetrain] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [autoRun, setAutoRun] = useState(() => localStorage.getItem(LS_AUTORUN) === '1')
  const [confirmClear, setConfirmClear] = useState(false)

  const job = useJob(jobId)
  const { recents, addRecent } = useRecentSymbols()

  const persist = (sym: string, name: string, algo: string) => {
    localStorage.setItem(LS_SYMBOL, sym)
    localStorage.setItem(LS_COMPANY, name)
    localStorage.setItem(LS_ALGORITHM, algo)
  }

  const handleAlgorithmChange = (algo: string) => {
    setAlgorithm(algo)
    localStorage.setItem(LS_ALGORITHM, algo)
  }

  const toggleAutoRun = (v: boolean) => {
    setAutoRun(v)
    localStorage.setItem(LS_AUTORUN, v ? '1' : '0')
  }

  const handleSymbol = (sym: string, name: string) => {
    setSymbol(sym)
    setCompany(name)
    setJobId(null)
    setRegimeResult(null)
    persist(sym, name, algorithm)
    addRecent(sym, name)
    if (autoRun && sym.trim()) {
      // Defer to next tick so state updates above are committed first
      setTimeout(() => runForecast(sym), 0)
    }
  }

  const runForecast = async (sym: string) => {
    try {
      const id = await startForecast(sym, algorithm, forceRetrain)
      setJobId(id)
      setTab('forecast')
      toast.success(
        forceRetrain
          ? `Retraining ${sym} from scratch`
          : `Job started for ${sym}`,
      )
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Failed to start job')
    }
  }

  const handleRun = async () => {
    if (!symbol.trim()) { toast.error('Please select a symbol first'); return }
    setJobId(null)
    setRegimeResult(null)
    await runForecast(symbol)
  }

  const handleRegime = async () => {
    if (!symbol.trim()) { toast.error('Please select a symbol first'); return }
    setRegimeLoading(true)
    setRegimeResult(null)
    setJobId(null)          // clear any previous forecast so charts don't conflict
    try {
      const res = await fetchRegimeAnalysis(symbol)
      setRegimeResult(res)
      setTab('regime')
      toast.success('Regime analysis complete')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Regime analysis failed')
    } finally { setRegimeLoading(false) }
  }

  const handleClearCache = () => {
    if (!symbol.trim()) { toast.error('Please select a symbol first'); return }
    setConfirmClear(true)
  }

  const confirmClearCache = async () => {
    setConfirmClear(false)
    setClearing(true)
    try {
      const res = await deleteCachedModels(symbol)
      toast.success(`Cleared ${res.deleted} cached model(s) for ${symbol}`)
    } catch (e: any) {
      toast.error(e?.response?.status === 404
        ? `No cached models for ${symbol}`
        : 'Failed to clear cache')
    } finally { setClearing(false) }
  }

  const isRunning = job?.status === 'pending' || job?.status === 'running'
  const isDone    = job?.status === 'done'
  const isError   = job?.status === 'error'

  const result         = job?.result as SingleForecastResult | AllForecastResult | null
  const isAll          = result?.algorithm === 'all'
  const singleResult   = !isAll ? result as SingleForecastResult : null
  const allResult      = isAll  ? result as AllForecastResult    : null

  const TABS: { id: TabId; label: string }[] = [
    { id: 'forecast', label: 'Forecast'       },
    { id: 'chart',    label: 'Chart'          },
    { id: 'regime',   label: 'Regime & Patterns' },
  ]

  return (
    <div className="max-w-screen-xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-6">

      <ConfirmDialog
        open={confirmClear}
        title="Clear saved weights?"
        message={`This will permanently delete the cached model weights for ${symbol}. The next forecast will retrain from scratch.`}
        confirmLabel="Clear weights"
        danger
        onConfirm={confirmClearCache}
        onCancel={() => setConfirmClear(false)}
      />

      {/* ── Left sidebar ────────────────────────────────────────────────── */}
      <aside className="space-y-5">
        <div className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-xl p-4 space-y-4">
          <SymbolSearch value={symbol} onChange={handleSymbol} onEnter={handleRun} />

          {/* Recent symbols */}
          {recents.length > 0 && (
            <div>
              <p className="flex items-center gap-1 text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5">
                <History className="w-3 h-3" /> Recent
              </p>
              <div className="flex flex-wrap gap-1.5">
                {recents.map(r => (
                  <button
                    key={r.symbol}
                    onClick={() => handleSymbol(r.symbol, r.name)}
                    title={r.name}
                    className={`px-2 py-1 rounded-md text-xs font-mono border transition ${
                      symbol === r.symbol
                        ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400 bg-indigo-500/10'
                        : 'border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-600'
                    }`}
                  >
                    {r.symbol}
                  </button>
                ))}
              </div>
            </div>
          )}

          <AlgorithmSelector value={algorithm} onChange={handleAlgorithmChange} />

          {/* Force retrain toggle */}
          <label className="flex items-start gap-2.5 cursor-pointer select-none group">
            <input
              type="checkbox"
              checked={forceRetrain}
              onChange={e => setForceRetrain(e.target.checked)}
              disabled={isRunning || regimeLoading}
              className="mt-0.5 w-4 h-4 shrink-0 rounded border-slate-400 dark:border-slate-600 bg-white dark:bg-slate-800
                         text-indigo-500 focus:ring-indigo-500 focus:ring-offset-white dark:focus:ring-offset-slate-900
                         disabled:opacity-40 cursor-pointer"
            />
            <span className="text-xs leading-tight">
              <span className="font-medium text-slate-600 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white transition">
                Force retrain
              </span>
              <span className="block text-slate-500 mt-0.5">
                Ignore saved weights and train from scratch
              </span>
            </span>
          </label>

          {/* Auto-run toggle */}
          <label className="flex items-start gap-2.5 cursor-pointer select-none group">
            <input
              type="checkbox"
              checked={autoRun}
              onChange={e => toggleAutoRun(e.target.checked)}
              className="mt-0.5 w-4 h-4 shrink-0 rounded border-slate-400 dark:border-slate-600 bg-white dark:bg-slate-800
                         text-indigo-500 focus:ring-indigo-500 focus:ring-offset-white dark:focus:ring-offset-slate-900
                         cursor-pointer"
            />
            <span className="text-xs leading-tight">
              <span className="font-medium text-slate-600 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white transition">
                Auto-run on select
              </span>
              <span className="block text-slate-500 mt-0.5">
                Start forecasting as soon as a symbol is picked
              </span>
            </span>
          </label>

          <button
            onClick={handleRun}
            disabled={isRunning || regimeLoading || !symbol}
            className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500
                       disabled:opacity-40 disabled:cursor-not-allowed
                       text-sm font-semibold text-white transition
                       flex items-center justify-center gap-2"
          >
            {forceRetrain ? <RotateCcw className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning
              ? (forceRetrain ? 'Retraining…' : 'Running…')
              : (forceRetrain ? 'Retrain & Forecast' : 'Run Forecast')}
          </button>

          <button
            onClick={handleRegime}
            disabled={isRunning || regimeLoading || !symbol}
            className="w-full py-2.5 rounded-lg bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600
                       disabled:opacity-40 disabled:cursor-not-allowed
                       text-sm font-medium text-slate-700 dark:text-slate-200 transition
                       flex items-center justify-center gap-2"
          >
            <Search className="w-4 h-4" />
            {regimeLoading ? 'Analysing…' : 'Regime & Pattern Analysis'}
          </button>

          <button
            onClick={handleClearCache}
            disabled={isRunning || regimeLoading || clearing || !symbol}
            className="w-full py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 hover:border-red-500/40
                       hover:text-red-500 dark:hover:text-red-400 disabled:opacity-40 disabled:cursor-not-allowed
                       text-xs font-medium text-slate-500 transition
                       flex items-center justify-center gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {clearing ? 'Clearing…' : `Clear saved weights${symbol ? ` for ${symbol}` : ''}`}
          </button>
        </div>

        {/* Status card */}
        {job && (
          <div
            className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-xl p-4 space-y-2 text-xs"
            role="status"
            aria-live="polite"
          >
            <p className="text-slate-500 uppercase font-semibold tracking-wider">Job Status</p>
            <p className="font-mono text-indigo-500 dark:text-indigo-400">{job.description}</p>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                job.status === 'done'    ? 'bg-emerald-400' :
                job.status === 'error'   ? 'bg-red-400'     :
                'bg-amber-400 animate-pulse'
              }`} />
              <span className="capitalize text-slate-700 dark:text-slate-300">{job.status}</span>
            </div>
            <p className="text-slate-500">{job.progress}</p>
          </div>
        )}
      </aside>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <main className="min-w-0">
        {/* Title bar */}
        {(result || regimeResult) && (
          <div className="mb-4 flex items-center justify-between gap-4 flex-wrap">
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">
              {result?.company_name ?? regimeResult?.company_name}{' '}
              <span className="text-slate-500 font-mono text-base">({symbol})</span>
            </h2>
            {isDone && result && (
              <CacheBadge
                status={result.cache_status}
                label={result.cache_label}
                info={(result as SingleForecastResult).cache_info}
              />
            )}
          </div>
        )}

        {/* Tabs */}
        {(result || regimeResult) && (
          <div className="flex gap-1 mb-4 border-b border-slate-300 dark:border-slate-800">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-2 text-sm font-medium transition border-b-2 -mb-px ${
                  tab === t.id
                    ? 'border-indigo-500 text-slate-900 dark:text-white'
                    : 'border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}

        {/* Loading */}
        {isRunning && (
          tab === 'chart'
            ? <SkeletonChart />
            : <SkeletonTable rows={5} />
        )}
        {regimeLoading && <LoadingSpinner message="Analysing regime & patterns…" />}

        {/* Error */}
        {isError && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-500 dark:text-red-400 text-sm">
            <strong>Error:</strong> {job?.error}
          </div>
        )}

        {/* ── Forecast tab ───────────────────────────────────────────────── */}
        {isDone && result && tab === 'forecast' && (
          <div className="space-y-4 animate-fadein" key={`forecast-${symbol}-${algorithm}`}>
            <ErrorBoundary label="Could not render the forecast table.">
              {singleResult && (
                <ForecastTable
                  forecast={singleResult.forecast}
                  displayName={singleResult.display_name}
                  rmse={singleResult.rmse}
                />
              )}
              {allResult && (
                <AllAlgosTable
                  algoForecasts={allResult.algo_forecasts}
                  allRmse={allResult.all_rmse}
                  bestAlgo={allResult.best_algo}
                  regime={allResult.regime}
                />
              )}
            </ErrorBoundary>
          </div>
        )}

        {/* ── Chart tab ──────────────────────────────────────────────────── */}
        {isDone && result && tab === 'chart' && (
          <div className="space-y-4 animate-fadein" key={`chart-${symbol}-${algorithm}`}>
            <ErrorBoundary label="Could not render the chart.">
              {singleResult && (
                <ForecastChart
                  historical={singleResult.historical}
                  forecast={singleResult.forecast}
                  algoLabel={singleResult.display_name}
                />
              )}
              {allResult && (
                // Show chart for the best algorithm in "all" mode
                <ForecastChart
                  historical={allResult.historical}
                  forecast={allResult.algo_forecasts[allResult.best_algo] ?? []}
                  algoLabel={`${allResult.best_algo.toUpperCase()} (Best)`}
                />
              )}
            </ErrorBoundary>
            {/* Regime mini card in chart view */}
            {result.regime && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <RegimeCard regime={result.regime} />
              </div>
            )}
          </div>
        )}

        {/* ── Regime tab (from forecast result — only when no standalone regime) */}
        {isDone && result && tab === 'regime' && result.regime && !regimeResult && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 animate-fadein">
            <RegimeCard regime={result.regime} />
          </div>
        )}

        {/* ── Regime tab (from standalone analysis) ─────────────────────── */}
        {regimeResult && tab === 'regime' && (
          <div className="space-y-4 animate-fadein" key={`regime-${symbol}`}>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <RegimeCard regime={regimeResult.regime} />
              <PatternList
                patterns={regimeResult.patterns}
                insight={regimeResult.insight}
              />
            </div>
            {regimeResult.historical.length > 0 && (
              <ErrorBoundary label="Could not render the chart.">
                <ForecastChart
                  historical={regimeResult.historical}
                  forecast={[]}
                  algoLabel="Historical"
                />
              </ErrorBoundary>
            )}
          </div>
        )}

        {/* ── Welcome / empty state ──────────────────────────────────────── */}
        {!job && !regimeResult && !regimeLoading && (
          <div className="flex flex-col items-center justify-center py-24 text-center text-slate-400 dark:text-slate-600">
            <LineChart className="w-12 h-12 mb-4 text-slate-300 dark:text-slate-700" />
            <h3 className="text-lg font-semibold text-slate-600 dark:text-slate-400 mb-2">Ready to Forecast</h3>
            <p className="text-sm max-w-md">
              Search for an NSE symbol on the left, select an algorithm, and click{' '}
              <span className="text-indigo-500 dark:text-indigo-400 font-medium">Run Forecast</span>.<br />
              Or use <span className="text-indigo-500 dark:text-indigo-400 font-medium">Regime & Pattern Analysis</span> first.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}





