import type { RegimeInfo } from '../types'

const REGIME_CFG = {
  BULL:     { emoji: '🟢', color: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/5'  },
  BEAR:     { emoji: '🔴', color: 'text-red-400',     border: 'border-red-500/30',     bg: 'bg-red-500/5'      },
  SIDEWAYS: { emoji: '🟡', color: 'text-amber-400',   border: 'border-amber-500/30',   bg: 'bg-amber-500/5'    },
  UNKNOWN:  { emoji: '⚪', color: 'text-slate-400',   border: 'border-slate-700',       bg: 'bg-slate-800/50'   },
}

interface Props {
  regime: RegimeInfo
}

export default function RegimeCard({ regime }: Props) {
  const cfg = REGIME_CFG[regime.regime] ?? REGIME_CFG.UNKNOWN

  return (
    <div className={`rounded-xl border ${cfg.border} ${cfg.bg} p-4`}>
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
        Market Regime
      </h3>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-2xl">{cfg.emoji}</span>
        <span className={`text-xl font-bold ${cfg.color}`}>{regime.regime}</span>
      </div>

      {regime.sufficient_data ? (
        <div className="space-y-1.5 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-500">SMA {50}</span>
            <span className="font-mono text-slate-300">{regime.sma_fast?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">SMA {200}</span>
            <span className="font-mono text-slate-300">{regime.sma_slow?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between pt-1 border-t border-slate-700/50">
            <span className="text-slate-500">Recommended</span>
            <span className="font-semibold text-indigo-400">{regime.recommended_model}</span>
          </div>
        </div>
      ) : (
        <p className="text-xs text-slate-500">{regime.description}</p>
      )}
    </div>
  )
}

