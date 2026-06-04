import type { SignalInfo } from '../types'

const CFG: Record<string, { bg: string; text: string; dot: string }> = {
  BUY:  { bg: 'bg-emerald-500/15', text: 'text-emerald-400', dot: 'bg-emerald-400' },
  SELL: { bg: 'bg-red-500/15',     text: 'text-red-400',     dot: 'bg-red-400'     },
  HOLD: { bg: 'bg-amber-500/15',   text: 'text-amber-400',   dot: 'bg-amber-400'   },
}

export default function SignalBadge({ signal }: { signal: SignalInfo | null }) {
  if (!signal) return <span className="text-slate-600 text-xs">—</span>

  const c = CFG[signal.label] ?? CFG['HOLD']

  const confStr = signal.regime_adjusted && signal.confidence_orig != null
    ? `${signal.confidence_orig}% ${signal.confidence_delta} = ${signal.confidence}%`
    : `${signal.confidence}%`

  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-semibold ${c.bg} ${c.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {signal.label}
      <span className="opacity-70 font-normal">{confStr}</span>
      {signal.regime_adjusted && (
        <span className="opacity-60 font-normal">{signal.regime_direction}</span>
      )}
    </span>
  )
}

