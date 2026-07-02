import type { PatternInfo } from '../types'

interface Props {
  patterns: PatternInfo[]
  insight:  string | null
}

export default function PatternList({ patterns, insight }: Props) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-4">
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
        Candlestick Patterns
        <span className="ml-2 text-slate-600 font-normal normal-case">(last 10 sessions)</span>
      </h3>

      {patterns.length === 0 ? (
        <p className="text-sm text-slate-500">No significant patterns detected</p>
      ) : (
        <ul className="space-y-2">
          {patterns.map((p, i) => (
            <li key={i} className="flex items-center justify-between text-sm">
              <span className="text-slate-300">{p.name}</span>
              <div className="flex items-center gap-2">
                <span className={p.direction === 'Bullish' ? 'text-emerald-400' : 'text-red-400'}>
                  {p.direction === 'Bullish' ? '🔼' : '🔽'} {p.direction}
                </span>
                <span className="text-xs text-slate-600">{p.date}</span>
              </div>
            </li>
          ))}
        </ul>
      )}

      {insight && (
        <div className="mt-3 pt-3 border-t border-slate-700/50 text-xs text-slate-300 leading-relaxed">
          {insight}
        </div>
      )}
    </div>
  )
}

