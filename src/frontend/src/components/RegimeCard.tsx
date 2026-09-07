import type { RegimeInfo } from '../types'
import { TrendingUp, TrendingDown, Minus, HelpCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

const REGIME_CFG: Record<string, { icon: LucideIcon; color: string; border: string; bg: string }> = {
  BULL:     { icon: TrendingUp,   color: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/5'  },
  BEAR:     { icon: TrendingDown, color: 'text-red-600 dark:text-red-400',         border: 'border-red-500/30',     bg: 'bg-red-500/5'      },
  SIDEWAYS: { icon: Minus,        color: 'text-amber-600 dark:text-amber-400',     border: 'border-amber-500/30',   bg: 'bg-amber-500/5'    },
  UNKNOWN:  { icon: HelpCircle,   color: 'text-slate-500 dark:text-slate-400',     border: 'border-slate-300 dark:border-slate-700', bg: 'bg-slate-100 dark:bg-slate-800/50' },
}

interface Props {
  regime: RegimeInfo
}

export default function RegimeCard({ regime }: Props) {
  const cfg = REGIME_CFG[regime.regime] ?? REGIME_CFG.UNKNOWN
  const Icon = cfg.icon

  return (
    <div className={`rounded-xl border ${cfg.border} ${cfg.bg} p-4`}>
      <h3 className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
        Market Regime
      </h3>
      <div className="flex items-center gap-2 mb-3">
        <Icon className={`w-6 h-6 ${cfg.color}`} />
        <span className={`text-xl font-bold ${cfg.color}`}>{regime.regime}</span>
      </div>

      {regime.sufficient_data ? (
        <div className="space-y-1.5 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-500">SMA {50}</span>
            <span className="font-mono text-slate-700 dark:text-slate-300">{regime.sma_fast?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">SMA {200}</span>
            <span className="font-mono text-slate-700 dark:text-slate-300">{regime.sma_slow?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between pt-1 border-t border-slate-300 dark:border-slate-700/50">
            <span className="text-slate-500">Recommended</span>
            <span className="font-semibold text-indigo-500 dark:text-indigo-400">{regime.recommended_model}</span>
          </div>
        </div>
      ) : (
        <p className="text-xs text-slate-500">{regime.description}</p>
      )}
    </div>
  )
}

