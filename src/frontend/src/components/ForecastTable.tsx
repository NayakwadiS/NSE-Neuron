import type { ForecastDay } from '../types'
import SignalBadge from './SignalBadge'

interface Props {
  forecast:    ForecastDay[]
  displayName: string
  rmse?:       number
}

const fmt = (v: number) => v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

export default function ForecastTable({ forecast, displayName, rmse }: Props) {
  const hasSignals = forecast.some(d => d.signal !== null)

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/40 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">
          {displayName} — 5-Day Forecast
        </h3>
        {rmse !== undefined && (
          <span className="text-xs font-mono text-slate-400">
            RMSE <span className="text-indigo-400">{rmse.toFixed(6)}</span>
          </span>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/50">
              {['Date', 'High', 'Low', 'Close', 'Prev Close', ...(hasSignals ? ['Signal'] : [])].map(h => (
                <th key={h} className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {forecast.map((day, i) => (
              <tr key={i} className="border-b border-slate-800 hover:bg-slate-700/20 transition">
                <td className="px-4 py-2.5 font-mono text-slate-300 text-xs">{day.date}</td>
                <td className="px-4 py-2.5 font-mono text-emerald-400">{fmt(day.high)}</td>
                <td className="px-4 py-2.5 font-mono text-red-400">{fmt(day.low)}</td>
                <td className="px-4 py-2.5 font-mono font-semibold text-white">{fmt(day.close)}</td>
                <td className="px-4 py-2.5 font-mono text-slate-400">{fmt(day.prev_close)}</td>
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
                <tr key={label} className="border-t border-slate-700 bg-slate-900/40">
                  <td className="px-4 py-2 text-xs font-semibold text-slate-500">{label}</td>
                  <td className="px-4 py-2 text-xs font-mono text-emerald-500">{fmt(vals.high)}</td>
                  <td className="px-4 py-2 text-xs font-mono text-red-500">{fmt(vals.low)}</td>
                  <td className="px-4 py-2 text-xs font-mono text-slate-300">{fmt(vals.close)}</td>
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

