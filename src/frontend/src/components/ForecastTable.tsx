import type { ForecastDay } from '../types'
import SignalBadge from './SignalBadge'
import { downloadCsv } from '../utils/export'
import { Download } from 'lucide-react'

interface Props {
  forecast:    ForecastDay[]
  displayName: string
  rmse?:       number
}

const fmt = (v: number) => v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

export default function ForecastTable({ forecast, displayName, rmse }: Props) {
  const hasSignals = forecast.some(d => d.signal !== null)

  const handleExport = () => {
    downloadCsv(
      `${displayName.replace(/\s+/g, '_')}_forecast.csv`,
      forecast.map(d => ({
        date: d.date,
        high: d.high,
        low: d.low,
        close: d.close,
        prev_close: d.prev_close,
        signal: d.signal?.label ?? '',
        confidence: d.signal?.confidence ?? '',
      })),
    )
  }

  return (
    <div className="rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800/40 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-300 dark:border-slate-700 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
          {displayName} — 5-Day Forecast
        </h3>
        <div className="flex items-center gap-3">
          {rmse !== undefined && (
            <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
              RMSE <span className="text-indigo-500 dark:text-indigo-400">{rmse.toFixed(6)}</span>
            </span>
          )}
          <button
            onClick={handleExport}
            title="Export as CSV"
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-indigo-500 dark:hover:text-indigo-400 transition text-slate-500 dark:text-slate-400"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-slate-50 dark:bg-slate-800">
            <tr className="border-b border-slate-300 dark:border-slate-700/50">
              {['Date', 'High', 'Low', 'Close', 'Prev Close', ...(hasSignals ? ['Signal'] : [])].map(h => (
                <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 dark:text-slate-500 uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {forecast.map((day, i) => (
              <tr key={i} className="border-b border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700/20 transition">
                <td className="px-4 py-2.5 font-mono text-slate-700 dark:text-slate-300 text-xs">{day.date}</td>
                <td className="px-4 py-2.5 font-mono text-emerald-600 dark:text-emerald-400">{fmt(day.high)}</td>
                <td className="px-4 py-2.5 font-mono text-red-600 dark:text-red-400">{fmt(day.low)}</td>
                <td className="px-4 py-2.5 font-mono font-semibold text-slate-900 dark:text-white">{fmt(day.close)}</td>
                <td className="px-4 py-2.5 font-mono text-slate-500 dark:text-slate-400">{fmt(day.prev_close)}</td>
                {hasSignals && (
                  <td className="px-4 py-2.5">
                    <SignalBadge signal={day.signal} />
                  </td>
                )}
              </tr>
            ))}
          </tbody>
          {/* Min/Max summary */}
          <tfoot>
            {(['Min', 'Max'] as const).map(label => {
              const vals = label === 'Min'
                ? {
                    high:  Math.min(...forecast.map(d => d.high)),
                    low:   Math.min(...forecast.map(d => d.low)),
                    close: Math.min(...forecast.map(d => d.close)),
                    prev:  Math.min(...forecast.map(d => d.prev_close)),
                  }
                : {
                    high:  Math.max(...forecast.map(d => d.high)),
                    low:   Math.max(...forecast.map(d => d.low)),
                    close: Math.max(...forecast.map(d => d.close)),
                    prev:  Math.max(...forecast.map(d => d.prev_close)),
                  }
              return (
                <tr key={label} className="border-t border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-900/40">
                  <td className="px-4 py-2 text-xs font-semibold text-slate-500">{label}</td>
                  <td className="px-4 py-2 text-xs font-mono text-emerald-600 dark:text-emerald-500">{fmt(vals.high)}</td>
                  <td className="px-4 py-2 text-xs font-mono text-red-600 dark:text-red-500">{fmt(vals.low)}</td>
                  <td className="px-4 py-2 text-xs font-mono text-slate-700 dark:text-slate-300">{fmt(vals.close)}</td>
                  <td className="px-4 py-2 text-xs font-mono text-slate-500">{fmt(vals.prev)}</td>
                  {hasSignals && <td className="px-4 py-2" />}
                </tr>
              )
            })}
          </tfoot>
        </table>
      </div>
    </div>
  )
}

