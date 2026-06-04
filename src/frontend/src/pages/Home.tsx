import { useState } from 'react'
import { startForecast, fetchRegimeAnalysis } from '../api/client'
import { useJob } from '../hooks/useJob'
import SymbolSearch       from '../components/SymbolSearch'
import AlgorithmSelector  from '../components/AlgorithmSelector'
import LoadingSpinner     from '../components/LoadingSpinner'
import ForecastTable      from '../components/ForecastTable'
import AllAlgosTable      from '../components/AllAlgosTable'
import ForecastChart      from '../components/ForecastChart'
import RegimeCard         from '../components/RegimeCard'
import PatternList        from '../components/PatternList'
import toast              from 'react-hot-toast'
import type {
  SingleForecastResult,
  AllForecastResult,
  RegimeAnalysisResult,
} from '../types'

type TabId = 'forecast' | 'chart' | 'regime'

export default function Home() {
  const [symbol,    setSymbol]    = useState('')
  const [company,   setCompany]   = useState('')
  const [algorithm, setAlgorithm] = useState('lstm')
  const [jobId,     setJobId]     = useState<string | null>(null)
  const [tab,       setTab]       = useState<TabId>('forecast')
  const [regimeResult, setRegimeResult] = useState<RegimeAnalysisResult | null>(null)
  const [regimeLoading, setRegimeLoading] = useState(false)

  const job = useJob(jobId)

  const handleSymbol = (sym: string, name: string) => {
    setSymbol(sym)
    setCompany(name)
    setJobId(null)
    setRegimeResult(null)
  }

  const handleRun = async () => {
    if (!symbol.trim()) { toast.error('Please select a symbol first'); return }
    setJobId(null)
    setRegimeResult(null)
    try {
      const id = await startForecast(symbol, algorithm)
      setJobId(id)
      setTab('forecast')
      toast.success(`Job started for ${symbol}`)
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Failed to start job')
    }
  }

  const handleRegime = async () => {
    if (!symbol.trim()) { toast.error('Please select a symbol first'); return }
    setRegimeLoading(true)
    setRegimeResult(null)
    try {
      const res = await fetchRegimeAnalysis(symbol)
      setRegimeResult(res)
      setTab('regime')
      toast.success('Regime analysis complete')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail ?? 'Regime analysis failed')
    } finally { setRegimeLoading(false) }
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
    <div className="max-w-screen-xl mx-auto px-4 py-6 grid grid-cols-[280px_1fr] gap-6">

      {/* ── Left sidebar ────────────────────────────────────────────────── */}
      <aside className="space-y-5">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4">
          <SymbolSearch value={symbol} onChange={handleSymbol} />
          <AlgorithmSelector value={algorithm} onChange={setAlgorithm} />

          <button
            onClick={handleRun}
            disabled={isRunning || regimeLoading || !symbol}
            className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500
                       disabled:opacity-40 disabled:cursor-not-allowed
                       text-sm font-semibold text-white transition"
          >
            {isRunning ? 'Training…' : '▶  Run Forecast'}
          </button>

          <button
            onClick={handleRegime}
            disabled={isRunning || regimeLoading || !symbol}
            className="w-full py-2.5 rounded-lg bg-slate-700 hover:bg-slate-600
                       disabled:opacity-40 disabled:cursor-not-allowed
                       text-sm font-medium text-slate-200 transition"
          >
            {regimeLoading ? 'Analysing…' : '🔍  Regime & Pattern Analysis'}
          </button>
        </div>

        {/* Status card */}
        {job && (
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2 text-xs">
            <p className="text-slate-500 uppercase font-semibold tracking-wider">Job Status</p>
            <p className="font-mono text-indigo-400">{job.description}</p>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${
                job.status === 'done'    ? 'bg-emerald-400' :
                job.status === 'error'   ? 'bg-red-400'     :
                'bg-amber-400 animate-pulse'
              }`} />
              <span className="capitalize text-slate-300">{job.status}</span>
            </div>
            <p className="text-slate-500">{job.progress}</p>
          </div>
        )}
      </aside>

      {/* ── Main content ─────────────────────────────────────────────────── */}
      <main className="min-w-0">
        {/* Title bar */}
        {(result || regimeResult) && (
          <div className="mb-4">
            <h2 className="text-lg font-bold text-white">
              {result?.company_name ?? regimeResult?.company_name}{' '}
              <span className="text-slate-500 font-mono text-base">({symbol})</span>
            </h2>
          </div>
        )}

        {/* Tabs */}
        {(result || regimeResult) && (
          <div className="flex gap-1 mb-4 border-b border-slate-800">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-2 text-sm font-medium transition border-b-2 -mb-px ${
                  tab === t.id
                    ? 'border-indigo-500 text-white'
                    : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}

        {/* Loading */}
        {isRunning && <LoadingSpinner message={job?.progress} />}
        {regimeLoading && <LoadingSpinner message="Analysing regime & patterns…" />}

        {/* Error */}
        {isError && (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-red-400 text-sm">
            <strong>Error:</strong> {job?.error}
          </div>
        )}

        {/* ── Forecast tab ───────────────────────────────────────────────── */}
        {isDone && result && tab === 'forecast' && (
          <div className="space-y-4">
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
          </div>
        )}

        {/* ── Chart tab ──────────────────────────────────────────────────── */}
        {isDone && result && tab === 'chart' && (
          <div className="space-y-4">
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
            {/* Regime mini card in chart view */}
            {result.regime && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <RegimeCard regime={result.regime} />
              </div>
            )}
          </div>
        )}

        {/* ── Regime tab (from forecast result) ─────────────────────────── */}
        {isDone && result && tab === 'regime' && result.regime && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <RegimeCard regime={result.regime} />
          </div>
        )}

        {/* ── Regime tab (from standalone analysis) ─────────────────────── */}
        {regimeResult && tab === 'regime' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <RegimeCard regime={regimeResult.regime} />
              <PatternList
                patterns={regimeResult.patterns}
                insight={regimeResult.insight}
              />
            </div>
            {regimeResult.historical.length > 0 && (
              <ForecastChart
                historical={regimeResult.historical}
                forecast={[]}
                algoLabel="Historical"
              />
            )}
          </div>
        )}

        {/* ── Welcome / empty state ──────────────────────────────────────── */}
        {!job && !regimeResult && !regimeLoading && (
          <div className="flex flex-col items-center justify-center py-24 text-center text-slate-600">
            <div className="text-5xl mb-4">📈</div>
            <h3 className="text-lg font-semibold text-slate-400 mb-2">Ready to Forecast</h3>
            <p className="text-sm max-w-md">
              Search for an NSE symbol on the left, select an algorithm, and click{' '}
              <span className="text-indigo-400 font-medium">Run Forecast</span>.<br />
              Or use <span className="text-indigo-400 font-medium">Regime & Pattern Analysis</span> first.
            </p>
          </div>
        )}
      </main>
    </div>
  )
}

