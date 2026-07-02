import type { ForecastDay } from '../types'

const ALGO_DISPLAY: Record<string, string> = {
  lstm: 'LSTM', bilstm: 'BiLSTM', gru: 'GRU', cnn_lstm: 'CNN-LSTM',
}

interface Props {
  algoForecasts: Record<string, ForecastDay[]>
  allRmse:       Record<string, number>
  bestAlgo:      string
  regime?:       { recommended_model: string; sufficient_data: boolean }
}

const fmt = (v: number) => v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

export default function AllAlgosTable({ algoForecasts, allRmse, bestAlgo, regime }: Props) {
  const algos = Object.keys(algoForecasts)
  const days  = algoForecasts[algos[0]] ?? []

  return (
    <div className="space-y-4">
      {/* Comparison table */}
      <div className="rounded-xl border border-slate-700 bg-slate-800/40 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-white">Close Price Forecast — All Algorithms</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">Algorithm</th>
                {days.map((d, i) => (
                  <th key={i} className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500">
                    Day {i + 1}<br />
                    <span className="font-mono font-normal text-slate-600">{d.date}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {algos.map(algo => {
                const isRecommended = regime?.sufficient_data && regime.recommended_model.toLowerCase().replace('-','_') === algo
                return (
                  <tr key={algo} className={`border-b border-slate-800 hover:bg-slate-700/20 ${isRecommended ? 'bg-indigo-500/5' : ''}`}>
                    <td className="px-4 py-2.5 font-semibold text-slate-300 flex items-center gap-2">
                      {ALGO_DISPLAY[algo] ?? algo}
                      {isRecommended && (
                        <span className="text-xs bg-indigo-500/20 text-indigo-400 px-1.5 py-0.5 rounded">Recommended</span>
                      )}
                    </td>
                    {algoForecasts[algo].map((d, i) => (
                      <td key={i} className="px-4 py-2.5 font-mono text-white">{fmt(d.close)}</td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* RMSE Benchmark */}
      <div className="rounded-xl border border-slate-700 bg-slate-800/40 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-700">
          <h3 className="text-sm font-semibold text-white">Model Benchmark — RMSE</h3>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/50">
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">Algorithm</th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">RMSE</th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">Rank</th>
            </tr>
          </thead>
          <tbody>
            {algos
              .slice()
              .sort((a, b) => allRmse[a] - allRmse[b])
              .map((algo, rank) => (
                <tr key={algo} className="border-b border-slate-800 hover:bg-slate-700/20">
                  <td className="px-4 py-2.5 font-semibold text-slate-300">{ALGO_DISPLAY[algo] ?? algo}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-300">{allRmse[algo].toFixed(6)}</td>
                  <td className="px-4 py-2.5">
                    {rank === 0 ? (
                      <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">🏆 Best</span>
                    ) : (
                      <span className="text-xs text-slate-600">#{rank + 1}</span>
                    )}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

